from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from .compose import compose_states, composition_report, save_report
from .controller import ForwardFineTuner, SignedLogMaskState
from .data import (
    load_jsonl_examples,
    save_jsonl_examples,
    tiny_arithmetic_eval,
    tiny_arithmetic_train,
)
from .memory import ControllerMemoryStore


def _load_hf(model_name: str, *, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError("dtype must be auto, bf16, fp16, or fp32")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(torch.device(device))
    model.eval()
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    return model, tok


def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])


def _make_tuner(args) -> ForwardFineTuner:
    model, tok = _load_hf(args.model, device=args.device, dtype=args.dtype)
    return ForwardFineTuner(
        model,
        tok,
        gates=getattr(args, "gates", 5000),
        layers=getattr(args, "layers", "all"),
        max_log_gate=getattr(args, "max_log_gate", 0.05),
        hook_site=getattr(args, "hook_site", "layer_output"),
    )


def _split_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_weights(s: str | None, n: int) -> list[float]:
    if s is None or not s.strip():
        return [1.0] * n
    vals = [float(x) for x in s.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"expected {n} weights, got {len(vals)}")
    return vals


def _parse_param_filter(s: str | None) -> list[str] | None:
    if s is None or not str(s).strip():
        return None
    vals = [x.strip() for x in str(s).split(",") if x.strip()]
    return vals or None


def _memory_text_from_examples(examples) -> str:
    parts = []
    for ex in examples[:8]:
        parts.append((ex.prompt + " " + ex.completion).strip())
    return "\n".join(parts)


def _write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# ---------- normal controller commands ----------


def cmd_fit(args) -> None:
    examples = load_jsonl_examples(args.train)
    tuner = _make_tuner(args)
    before = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    fit_stats = tuner.fit(
        examples,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        score_batches=args.score_batches,
        max_length=args.max_length,
        verbose=not args.quiet,
    )
    after = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    tuner.save(args.out)
    manifest_path = str(Path(args.out).with_suffix(".manifest.json"))
    tuner.write_manifest(manifest_path)
    print(json.dumps({
        "controller": args.out,
        "manifest": manifest_path,
        "train_examples": len(examples),
        "before": before,
        "after": after,
        "fit": fit_stats,
    }, indent=2), flush=True)


def cmd_fit_dual(args) -> None:
    examples = load_jsonl_examples(args.train)
    tuner = _make_tuner(args)
    before = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    fit_stats = tuner.fit_dual(
        examples,
        steps=args.steps,
        target_step_size=args.target_step_size,
        apply_scale=args.apply_scale,
        batch_size=args.batch_size,
        score_batches=args.score_batches,
        max_length=args.max_length,
        projection=args.projection,
        top_k=args.top_k,
        ridge=args.ridge,
        cg_iters=args.cg_iters,
        cg_tol=args.cg_tol,
        fd_eps=args.fd_eps,
        metric=args.metric,
        metric_eps=args.metric_eps,
        jvp_mode=args.jvp_mode,
        param_name_substrings=_parse_param_filter(args.param_filter),
        max_target_parameters=args.max_target_parameters,
        verbose=not args.quiet,
    )
    after = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    tuner.save(args.out)
    manifest_path = str(Path(args.out).with_suffix(".manifest.json"))
    tuner.write_manifest(manifest_path)
    result = {
        "controller": args.out,
        "manifest": manifest_path,
        "train_examples": len(examples),
        "before": before,
        "after": after,
        "fit_dual": fit_stats,
    }
    if args.report:
        _write_json(args.report, result)
    print(json.dumps(result, indent=2), flush=True)


def cmd_secant(args) -> None:
    examples = load_jsonl_examples(args.eval)
    tuner = _make_tuner(args)
    tuner.load(args.controller)
    report = tuner.secant_diagnostics(
        examples,
        eps=args.fd_eps,
        batch_size=args.batch_size,
        max_length=args.max_length,
        projection=args.projection,
        top_k=args.top_k,
    )
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)


def cmd_dual_diagnose(args) -> None:
    support = load_jsonl_examples(args.support)
    calibration = load_jsonl_examples(args.calibration) if args.calibration else None
    tuner = _make_tuner(args)
    if args.controller:
        tuner.load(args.controller)
        init = {"source": args.controller}
    else:
        init = tuner.initialize_controller(
            support,
            score_batches=args.score_batches,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    report = tuner.dual_projection_diagnostics(
        support,
        calibration,
        batch_size=args.batch_size,
        max_length=args.max_length,
        projection=args.projection,
        top_k=args.top_k,
        target_step_size=args.target_step_size,
        ridge=args.ridge,
        cg_iters=args.cg_iters,
        cg_tol=args.cg_tol,
        fd_eps=args.fd_eps,
        metric=args.metric,
        metric_eps=args.metric_eps,
        jvp_mode=args.jvp_mode,
        param_name_substrings=_parse_param_filter(args.param_filter),
        max_target_parameters=args.max_target_parameters,
    )
    report["basis"] = init
    if args.out:
        _write_json(args.out, report)
    print(json.dumps(report, indent=2), flush=True)


def cmd_eval(args) -> None:
    examples = load_jsonl_examples(args.eval)
    tuner = _make_tuner(args)
    base = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    ctrl = None
    if args.controller:
        tuner.load(args.controller)
        ctrl = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    result = {"examples": len(examples), "base": base, "controller": ctrl}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2), flush=True)


def cmd_generate(args) -> None:
    tuner = _make_tuner(args)
    tuner.load(args.controller)
    print(tuner.generate(args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False), flush=True)


def cmd_demo(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "math_train.jsonl"
    eval_path = out_dir / "math_eval.jsonl"
    ctrl_path = out_dir / "controller.pt"
    save_jsonl_examples(train_path, tiny_arithmetic_train())
    save_jsonl_examples(eval_path, tiny_arithmetic_eval())
    print(f"wrote {train_path} and {eval_path}", flush=True)

    tuner = _make_tuner(args)
    train = load_jsonl_examples(train_path)
    eval_examples = load_jsonl_examples(eval_path)
    base_train = tuner.evaluate_nll(train, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    base_eval = tuner.evaluate_nll(eval_examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    fit_stats = tuner.fit(
        train,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        score_batches=args.score_batches,
        max_length=args.max_length,
        verbose=True,
    )
    ctrl_train = tuner.evaluate_nll(train, batch_size=args.batch_size, max_length=args.max_length)
    ctrl_eval = tuner.evaluate_nll(eval_examples, batch_size=args.batch_size, max_length=args.max_length)
    tuner.save(ctrl_path)
    print(json.dumps({
        "controller": str(ctrl_path),
        "train_base": base_train,
        "train_controller": ctrl_train,
        "eval_base": base_eval,
        "eval_controller": ctrl_eval,
        "fit": fit_stats,
    }, indent=2), flush=True)


def cmd_compose(args) -> None:
    states = [SignedLogMaskState.load(path, map_location="cpu") for path in args.controllers]
    weights = _parse_weights(args.weights, len(states))
    composed = compose_states(states, weights=weights, max_log_gate=args.max_log_gate)
    composed.save(args.out)
    report = composition_report(args.controllers, states)
    report["composition"] = {
        "out": args.out,
        "weights": weights,
        "n_gates": composed.n_gates,
        "max_log_gate": composed.max_log_gate,
    }
    if args.report:
        save_report(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


def cmd_inspect(args) -> None:
    states = [SignedLogMaskState.load(path, map_location="cpu") for path in args.controllers]
    report = composition_report(args.controllers, states)
    if args.out:
        save_report(args.out, report)
    print(json.dumps(report, indent=2), flush=True)


# ---------- persistent memory commands ----------


def cmd_memory_add(args) -> None:
    store = ControllerMemoryStore(args.store)
    tags = _split_csv(args.tags)
    meta = {"source": args.train or args.controller}
    if args.controller:
        if not args.text:
            raise ValueError("--text is required when registering an existing --controller")
        item = store.add_controller(
            memory_id=args.id,
            controller_path=args.controller,
            text=args.text,
            tags=tags,
            metadata=meta,
            overwrite=args.overwrite,
        )
        print(json.dumps({"added": item.id, "item": item.__dict__}, indent=2), flush=True)
        return

    if not args.train:
        raise ValueError("memory add requires either --train or --controller")
    examples = load_jsonl_examples(args.train)
    text = args.text or _memory_text_from_examples(examples)
    tuner = _make_tuner(args)
    before = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    fit_stats = tuner.fit(
        examples,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        score_batches=args.score_batches,
        max_length=args.max_length,
        verbose=not args.quiet,
    )
    after = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    tmp_controller = Path(args.store) / f".tmp-{args.id}.pt"
    tuner.save(tmp_controller)
    meta.update({"before": before, "after": after, "fit": fit_stats})
    try:
        item = store.add_controller(
            memory_id=args.id,
            controller_path=tmp_controller,
            text=text,
            tags=tags,
            metadata=meta,
            overwrite=True,
        )
    finally:
        try:
            tmp_controller.unlink()
        except OSError:
            pass
    print(json.dumps({"added": item.id, "before": before, "after": after, "fit": fit_stats}, indent=2), flush=True)


def cmd_memory_list(args) -> None:
    store = ControllerMemoryStore(args.store)
    print(json.dumps({"items": [x.__dict__ for x in store.list_items()]}, indent=2), flush=True)


def cmd_memory_delete(args) -> None:
    store = ControllerMemoryStore(args.store)
    store.delete(args.id)
    print(json.dumps({"deleted": args.id}, indent=2), flush=True)


def cmd_memory_search(args) -> None:
    store = ControllerMemoryStore(args.store)
    hits = store.search(args.query, top_k=args.top_k, tag=args.tag, min_score=args.min_score)
    hits = store.weight_hits(hits, weighting=args.weighting, temperature=args.temperature)
    print(json.dumps({"query": args.query, "hits": [h.to_dict() for h in hits]}, indent=2), flush=True)


def cmd_memory_compose(args) -> None:
    store = ControllerMemoryStore(args.store)
    state, hits = store.compose_for_query(
        args.query,
        top_k=args.top_k,
        weighting=args.weighting,
        temperature=args.temperature,
        max_log_gate=args.max_log_gate,
        tag=args.tag,
        min_score=args.min_score,
    )
    state.save(args.out)
    report = {
        "query": args.query,
        "out": args.out,
        "n_gates": state.n_gates,
        "max_log_gate": state.max_log_gate,
        "hits": [h.to_dict() for h in hits],
    }
    if args.report:
        _write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


def cmd_memory_generate(args) -> None:
    store = ControllerMemoryStore(args.store)
    state, hits = store.compose_for_query(
        args.query,
        top_k=args.top_k,
        weighting=args.weighting,
        temperature=args.temperature,
        max_log_gate=args.compose_max_log_gate,
        tag=args.tag,
        min_score=args.min_score,
    )
    tuner = _make_tuner(args)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)
    try:
        state.save(tmp)
        tuner.load(tmp)
        text = tuner.generate(args.prompt or args.query, max_new_tokens=args.max_new_tokens, do_sample=False)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    print(json.dumps({"query": args.query, "hits": [h.to_dict() for h in hits], "text": text}, indent=2), flush=True)


def cmd_memory_eval(args) -> None:
    examples = load_jsonl_examples(args.eval)
    store = ControllerMemoryStore(args.store)
    state, hits = store.compose_for_query(
        args.query,
        top_k=args.top_k,
        weighting=args.weighting,
        temperature=args.temperature,
        max_log_gate=args.compose_max_log_gate,
        tag=args.tag,
        min_score=args.min_score,
    )
    tuner = _make_tuner(args)
    base = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length, use_controller=False)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)
    try:
        state.save(tmp)
        tuner.load(tmp)
        ctrl = tuner.evaluate_nll(examples, batch_size=args.batch_size, max_length=args.max_length)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    result = {"query": args.query, "hits": [h.to_dict() for h in hits], "base": base, "controller": ctrl}
    if args.out:
        _write_json(args.out, result)
    print(json.dumps(result, indent=2), flush=True)


# ---------- parser ----------


def _add_tuner_args(p: argparse.ArgumentParser, *, demo: bool = False) -> None:
    p.add_argument("--gates", type=int, default=1024 if demo else 5000)
    p.add_argument("--steps", type=int, default=80 if demo else 240)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=2 if demo else 8)
    p.add_argument("--score-batches", type=int, default=4 if demo else 16)
    p.add_argument("--layers", default="all")
    p.add_argument("--max-log-gate", type=float, default=0.05)
    p.add_argument("--hook-site", default="layer_output", choices=["layer_output", "layer_input"])
    p.add_argument("--max-length", type=int, default=512 if demo else 1024)


def _add_memory_retrieval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--tag", default=None)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--weighting", default="softmax", choices=["softmax", "score", "uniform"])
    p.add_argument("--temperature", type=float, default=0.25)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ntkmirror",
        description="LoRA-free signed log-mask forward fine-tuning for Hugging Face causal LMs.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit", help="fit a signed log-mask controller on JSONL examples")
    _add_common_model_args(fit)
    fit.add_argument("--train", required=True)
    fit.add_argument("--out", required=True)
    _add_tuner_args(fit)
    fit.add_argument("--quiet", action="store_true")
    fit.set_defaults(func=cmd_fit)

    fit_dual = sub.add_parser(
        "fit-dual",
        help="experimental pathwise activation-control NTK fit against full-weight SGD fields",
    )
    _add_common_model_args(fit_dual)
    fit_dual.add_argument("--train", required=True)
    fit_dual.add_argument("--out", required=True)
    fit_dual.add_argument("--report", default=None)
    _add_tuner_args(fit_dual)
    fit_dual.add_argument("--projection", default="target", choices=["target", "topk", "full"])
    fit_dual.add_argument("--top-k", type=int, default=32)
    fit_dual.add_argument("--target-step-size", type=float, default=1e-5)
    fit_dual.add_argument("--apply-scale", type=float, default=1.0)
    fit_dual.add_argument("--ridge", type=float, default=1e-4)
    fit_dual.add_argument("--cg-iters", type=int, default=16)
    fit_dual.add_argument("--cg-tol", type=float, default=1e-5)
    fit_dual.add_argument("--fd-eps", type=float, default=1e-3, help="legacy finite-difference epsilon; only used with --jvp-mode fd")
    fit_dual.add_argument("--jvp-mode", default="exact", choices=["exact", "fd"], help="exact uses autograd JVP; fd is a legacy diagnostic fallback")
    fit_dual.add_argument("--metric", default="identity", choices=["identity", "activation"], help="gate metric M in BM^-1B^T")
    fit_dual.add_argument("--metric-eps", type=float, default=1e-6)
    fit_dual.add_argument("--param-filter", default=None, help="comma substrings of model parameter names to include in the full-weight target")
    fit_dual.add_argument("--max-target-parameters", type=int, default=0, help="0 disables the guard; otherwise fail above this many differentiated parameters")
    fit_dual.add_argument("--quiet", action="store_true")
    fit_dual.set_defaults(func=cmd_fit_dual)

    sec = sub.add_parser("secant", help="diagnose finite controller departure from the initial gate tangent")
    _add_common_model_args(sec)
    sec.add_argument("--controller", required=True)
    sec.add_argument("--eval", required=True)
    sec.add_argument("--out", default=None)
    sec.add_argument("--batch-size", type=int, default=1)
    sec.add_argument("--max-length", type=int, default=1024)
    sec.add_argument("--projection", default="target", choices=["target", "topk", "full"])
    sec.add_argument("--top-k", type=int, default=32)
    sec.add_argument("--fd-eps", type=float, default=1e-3)
    sec.add_argument("--gates", type=int, default=5000)
    sec.add_argument("--layers", default="all")
    sec.add_argument("--max-log-gate", type=float, default=0.05)
    sec.add_argument("--hook-site", default="layer_output", choices=["layer_output", "layer_input"])
    sec.set_defaults(func=cmd_secant)

    diag = sub.add_parser(
        "dual-diagnose",
        help="test whether the full-weight SGD field lies in the selected gate tangent range",
    )
    _add_common_model_args(diag)
    diag.add_argument("--support", required=True)
    diag.add_argument("--calibration", default=None)
    diag.add_argument("--controller", default=None, help="existing controller basis; if omitted, score a fresh zero basis")
    diag.add_argument("--out", default=None)
    diag.add_argument("--gates", type=int, default=5000)
    diag.add_argument("--layers", default="all")
    diag.add_argument("--max-log-gate", type=float, default=0.05)
    diag.add_argument("--hook-site", default="layer_output", choices=["layer_output", "layer_input"])
    diag.add_argument("--score-batches", type=int, default=16)
    diag.add_argument("--batch-size", type=int, default=1)
    diag.add_argument("--max-length", type=int, default=1024)
    diag.add_argument("--projection", default="target", choices=["target", "topk", "full"])
    diag.add_argument("--top-k", type=int, default=32)
    diag.add_argument("--target-step-size", type=float, default=1e-5)
    diag.add_argument("--ridge", type=float, default=1e-4)
    diag.add_argument("--cg-iters", type=int, default=16)
    diag.add_argument("--cg-tol", type=float, default=1e-5)
    diag.add_argument("--fd-eps", type=float, default=1e-3, help="legacy finite-difference epsilon; only used with --jvp-mode fd")
    diag.add_argument("--jvp-mode", default="exact", choices=["exact", "fd"], help="exact uses autograd JVP; fd is a legacy diagnostic fallback")
    diag.add_argument("--metric", default="identity", choices=["identity", "activation"], help="gate metric M in BM^-1B^T")
    diag.add_argument("--metric-eps", type=float, default=1e-6)
    diag.add_argument("--param-filter", default=None, help="comma substrings of model parameter names to include in the full-weight target")
    diag.add_argument("--max-target-parameters", type=int, default=0, help="0 disables the guard; otherwise fail above this many differentiated parameters")
    diag.set_defaults(func=cmd_dual_diagnose)

    ev = sub.add_parser("eval", help="evaluate base and optionally controller NLL/token accuracy")
    _add_common_model_args(ev)
    ev.add_argument("--eval", required=True)
    ev.add_argument("--controller", default=None)
    ev.add_argument("--out", default=None)
    ev.add_argument("--batch-size", type=int, default=8)
    ev.add_argument("--max-length", type=int, default=1024)
    ev.set_defaults(func=cmd_eval)

    gen = sub.add_parser("generate", help="generate with a fitted controller attached")
    _add_common_model_args(gen)
    gen.add_argument("--controller", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-new-tokens", type=int, default=128)
    gen.add_argument("--gates", type=int, default=5000)
    gen.add_argument("--layers", default="all")
    gen.add_argument("--max-log-gate", type=float, default=0.05)
    gen.set_defaults(func=cmd_generate)

    demo = sub.add_parser("demo", help="write and run a tiny arithmetic demo")
    _add_common_model_args(demo)
    demo.add_argument("--out-dir", default="runs/demo")
    _add_tuner_args(demo, demo=True)
    demo.set_defaults(func=cmd_demo)

    comp = sub.add_parser("compose", help="compose controllers by adding signed log-gates")
    comp.add_argument("--controllers", nargs="+", required=True)
    comp.add_argument("--out", required=True)
    comp.add_argument("--weights", default=None)
    comp.add_argument("--max-log-gate", type=float, default=None)
    comp.add_argument("--report", default=None)
    comp.set_defaults(func=cmd_compose)

    ins = sub.add_parser("inspect", help="inspect controller size, overlap, and gate-space cosine")
    ins.add_argument("--controllers", nargs="+", required=True)
    ins.add_argument("--out", default=None)
    ins.set_defaults(func=cmd_inspect)

    mem = sub.add_parser("memory", help="persistent controller memory")
    mem_sub = mem.add_subparsers(dest="memory_cmd", required=True)

    mem_add = mem_sub.add_parser("add", help="fit or register one controller memory item")
    _add_common_model_args(mem_add)
    mem_add.add_argument("--store", required=True)
    mem_add.add_argument("--id", required=True)
    mem_add.add_argument("--train", default=None, help="JSONL used to fit a new controller")
    mem_add.add_argument("--controller", default=None, help="existing controller .pt to register")
    mem_add.add_argument("--text", default=None, help="retrieval text; defaults to first training examples")
    mem_add.add_argument("--tags", default=None, help="comma tags, e.g. math,user-pref")
    mem_add.add_argument("--overwrite", action="store_true")
    _add_tuner_args(mem_add)
    mem_add.add_argument("--quiet", action="store_true")
    mem_add.set_defaults(func=cmd_memory_add)

    mem_list = mem_sub.add_parser("list", help="list memory items")
    mem_list.add_argument("--store", required=True)
    mem_list.set_defaults(func=cmd_memory_list)

    mem_del = mem_sub.add_parser("delete", help="delete a memory item")
    mem_del.add_argument("--store", required=True)
    mem_del.add_argument("--id", required=True)
    mem_del.set_defaults(func=cmd_memory_delete)

    mem_search = mem_sub.add_parser("search", help="retrieve candidate controller memories")
    _add_memory_retrieval_args(mem_search)
    mem_search.set_defaults(func=cmd_memory_search)

    mem_comp = mem_sub.add_parser("compose", help="retrieve and compose controllers for a query")
    _add_memory_retrieval_args(mem_comp)
    mem_comp.add_argument("--out", required=True)
    mem_comp.add_argument("--max-log-gate", type=float, default=None)
    mem_comp.add_argument("--report", default=None)
    mem_comp.set_defaults(func=cmd_memory_compose)

    mem_gen = mem_sub.add_parser("generate", help="retrieve, compose, and generate with memory controllers")
    _add_common_model_args(mem_gen)
    _add_memory_retrieval_args(mem_gen)
    mem_gen.add_argument("--prompt", default=None, help="generation prompt; defaults to query")
    mem_gen.add_argument("--max-new-tokens", type=int, default=128)
    mem_gen.add_argument("--compose-max-log-gate", type=float, default=None)
    mem_gen.add_argument("--gates", type=int, default=5000)
    mem_gen.add_argument("--layers", default="all")
    mem_gen.add_argument("--max-log-gate", type=float, default=0.05)
    mem_gen.set_defaults(func=cmd_memory_generate)

    mem_eval = mem_sub.add_parser("eval", help="retrieve, compose, and evaluate NLL on JSONL examples")
    _add_common_model_args(mem_eval)
    _add_memory_retrieval_args(mem_eval)
    mem_eval.add_argument("--eval", required=True)
    mem_eval.add_argument("--compose-max-log-gate", type=float, default=None)
    mem_eval.add_argument("--batch-size", type=int, default=8)
    mem_eval.add_argument("--max-length", type=int, default=1024)
    mem_eval.add_argument("--gates", type=int, default=5000)
    mem_eval.add_argument("--layers", default="all")
    mem_eval.add_argument("--max-log-gate", type=float, default=0.05)
    mem_eval.add_argument("--out", default=None)
    mem_eval.set_defaults(func=cmd_memory_eval)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

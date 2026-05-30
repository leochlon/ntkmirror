#!/usr/bin/env python3
"""Safe diffusion scale-gate trainer with finite metric-aware self-pruning.

This script keeps the useful part of ``train_scale_gate_adam_M.py``:
activation-energy calibration, signed log-scale gates, Adam moments, and the
step-adaptive metric M_t = base_M * exp(2q). It removes the unsafe part: no
NaN/inf leak is allowed to bypass the global cap.

The important change is that pruning is represented explicitly and finitely:
channels whose q <= q_prune are treated as hard-dead in the forward pass and
are excluded from the active M-norm cap. Active channels are guarded by a
finite metric floor, per-coordinate delta caps, and non-finite update checks.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


HOOK_CLASSES = {"ResnetBlock2D", "BasicTransformerBlock", "Attention"}


def _finite_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.detach().float().reshape(-1)
    finite = torch.isfinite(xf)
    if not bool(finite.any()):
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "finite_frac": 0.0}
    vals = xf[finite]
    return {
        "min": float(vals.min().item()),
        "max": float(vals.max().item()),
        "mean": float(vals.mean().item()),
        "finite_frac": float(finite.float().mean().item()),
    }


class ScaleGateBank(nn.Module):
    def __init__(
        self,
        *,
        q_max: float = 20.0,
        q_prune: float = -20.0,
        shift_clip: float = 10.0,
        hard_prune_forward: bool = True,
        train_shift: bool = True,
    ):
        super().__init__()
        self.q_max = float(q_max)
        self.q_prune = float(q_prune)
        self.shift_clip = float(shift_clip)
        self.hard_prune_forward = bool(hard_prune_forward)
        self.train_shift = bool(train_shift)
        self.params = nn.ParameterDict()
        self.shifts = nn.ParameterDict()
        self.specs: dict[str, int] = {}
        self.enabled = True
        self._calib_mode = False
        self._calib_count: dict[str, int] = {}
        self._calib_sumsq: dict[str, torch.Tensor] = {}

    @staticmethod
    def key(name: str) -> str:
        return name.replace(".", "__")

    def ensure(self, name: str, out: torch.Tensor) -> None:
        key = self.key(name)
        if key in self.params:
            return
        if out.dim() == 4:
            c = int(out.shape[1])
        elif out.dim() == 3:
            c = int(out.shape[-1])
        else:
            return
        self.params[key] = nn.Parameter(torch.zeros(c, dtype=torch.float32, device=out.device))
        self.shifts[key] = nn.Parameter(torch.zeros(c, dtype=torch.float32, device=out.device), requires_grad=self.train_shift)
        self.specs[key] = c
        self._calib_count[key] = 0
        self._calib_sumsq[key] = torch.zeros(c, dtype=torch.float32, device=out.device)

    def _scale_from_q(self, q: torch.Tensor) -> torch.Tensor:
        q_safe = q.clamp(min=self.q_prune, max=self.q_max)
        scale = torch.exp(q_safe)
        if self.hard_prune_forward:
            scale = torch.where(q <= self.q_prune, torch.zeros_like(scale), scale)
        return scale

    def gate(self, name: str, out: torch.Tensor):
        self.ensure(name, out)
        key = self.key(name)
        if self._calib_mode and key in self._calib_sumsq:
            if out.dim() == 4:
                ss = (out.detach().float() ** 2).mean(dim=(0, 2, 3))
            elif out.dim() == 3:
                ss = (out.detach().float() ** 2).mean(dim=(0, 1))
            else:
                ss = None
            if ss is not None:
                self._calib_sumsq[key] += ss
                self._calib_count[key] += 1
        if not self.enabled or key not in self.params:
            return out
        q = self.params[key].to(dtype=out.dtype)
        scale = self._scale_from_q(q).to(dtype=out.dtype)
        if self.train_shift:
            b = self.shifts[key].clamp(-self.shift_clip, self.shift_clip).to(dtype=out.dtype)
        else:
            b = torch.zeros_like(self.shifts[key], dtype=out.dtype)
        if out.dim() == 4:
            return scale.view(1, -1, 1, 1) * out + b.view(1, -1, 1, 1)
        if out.dim() == 3:
            return scale.view(1, 1, -1) * out + b.view(1, 1, -1)
        return out

    def base_M_from_calib(self, eps: float = 1e-6) -> dict[str, torch.Tensor]:
        out = {}
        for k, ss in self._calib_sumsq.items():
            cnt = max(self._calib_count[k], 1)
            out[k] = (ss / cnt).clamp_min(float(eps)).detach().float()
        return out

    def q_summary(self) -> dict[str, float]:
        if not self.params:
            return {"max": 0.0, "min": 0.0, "dead_frac": 0.0, "finite_frac": 1.0}
        q = torch.cat([v.detach().float().reshape(-1).cpu() for v in self.params.values()])
        finite = torch.isfinite(q)
        dead = q <= self.q_prune
        return {
            "max": float(q[finite].max().item()) if bool(finite.any()) else math.nan,
            "min": float(q[finite].min().item()) if bool(finite.any()) else math.nan,
            "dead_frac": float(dead.float().mean().item()),
            "finite_frac": float(finite.float().mean().item()),
        }

    def save(self, path: str | Path, *, meta: dict, base_M: dict[str, torch.Tensor] | None = None) -> None:
        sd = {k: v.detach().float().cpu() for k, v in self.state_dict().items()}
        payload = {
            "state_dict": sd,
            "specs": self.specs,
            "meta": meta,
            "q_max": self.q_max,
            "q_prune": self.q_prune,
            "shift_clip": self.shift_clip,
            "hard_prune_forward": self.hard_prune_forward,
            "train_shift": self.train_shift,
        }
        if base_M is not None:
            payload["base_M"] = {k: v.detach().float().cpu() for k, v in base_M.items()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)


def install_gate(pipe, bank: ScaleGateBank) -> int:
    count = 0
    for name, module in pipe.unet.named_modules():
        if module.__class__.__name__ not in HOOK_CLASSES:
            continue
        def hook(_mod, _inp, out, name=name):
            if torch.is_tensor(out):
                return bank.gate(name, out)
            if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
                return (bank.gate(name, out[0]),) + out[1:]
            return out
        module.register_forward_hook(hook)
        count += 1
    pipe.unet.add_module("scale_gate_bank", bank)
    return count


def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < h:
        new_w = int(size)
        new_h = max(int(size), int(round(h * size / max(w, 1))))
    else:
        new_h = int(size)
        new_w = max(int(size), int(round(w * size / max(h, 1))))
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = max(0, (new_w - size) // 2)
    top = max(0, (new_h - size) // 2)
    return img.crop((left, top, left + size, top + size))


def _pil_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    w, h = img.size
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    x = data.view(h, w, 3).permute(2, 0, 1).contiguous().float().div_(255.0)
    return x.mul_(2.0).sub_(1.0)


def load_images(image_dir: str | Path, n: int, size: int, device: torch.device) -> torch.Tensor:
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths.extend(Path(image_dir).glob(ext))
    paths = sorted(paths)[: int(n)]
    if not paths:
        raise FileNotFoundError(f"no images found in {image_dir}")
    tensors = []
    for p in paths:
        img = _resize_center_crop(Image.open(p).convert("RGB"), int(size))
        tensors.append(_pil_to_normalized_tensor(img))
    return torch.stack(tensors).to(device=device, dtype=torch.float32)


def _parse_dtype(name: str):
    name = str(name).lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError("dtype must be fp32, fp16, or bf16")


def _active_metric(base_M: torch.Tensor, q: torch.Tensor, *, metric_floor: float, q_prune: float, m_damping: str = "cosh") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qf = q.detach().float()
    active = torch.isfinite(qf) & (qf > float(q_prune))
    qc = qf.clamp(min=-80.0, max=80.0) if m_damping == "cosh" else qf.clamp(min=float(q_prune), max=80.0)
    damping = torch.cosh(2.0 * qc) if m_damping == "cosh" else torch.exp(2.0 * qc)
    M_actual = base_M.detach().float().to(q.device) * damping
    M_eff = M_actual.clamp_min(float(metric_floor))
    return active, M_actual, M_eff


def _clip_signed_update(u: torch.Tensor, max_abs_update: float) -> torch.Tensor:
    if max_abs_update <= 0:
        return u
    return u.clamp(min=-float(max_abs_update), max=float(max_abs_update))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--tau-q", type=float, default=0.0, help="active q-update M-norm cap; 0 disables")
    ap.add_argument("--tau-b", type=float, default=0.0, help="shift-update Euclidean cap; 0 disables")
    ap.add_argument("--max-delta-q", type=float, default=1.0, help="max |lr * update_q| per step; 0 disables")
    ap.add_argument("--max-delta-b", type=float, default=0.25, help="max |lr * update_b| per step; 0 disables")
    ap.add_argument("--metric-floor", type=float, default=1e-6, help="floor for M_eff used in 1/sqrt(M_eff)")
    ap.add_argument("--q-max", type=float, default=20.0)
    ap.add_argument("--q-prune", type=float, default=-20.0)
    ap.add_argument("--shift-clip", type=float, default=10.0)
    ap.add_argument("--train-shift", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--hard-prune-forward", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch-n", type=int, default=2)
    ap.add_argument("--image-n", type=int, default=5)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-steps", type=int, default=8)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--m-damping", default="cosh", choices=["cosh", "exp"], help="self-damping in M_t: cosh symmetric (default, correct) or exp asymmetric (legacy, broken for q<0)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this trainer expects a CUDA device")
    device = torch.device("cuda")
    dtype = _parse_dtype(args.dtype)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler
    except ImportError as exc:
        raise RuntimeError("Install diffusion extras first: pip install -e '.[diffusion]'") from exc

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    train_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
    for p in pipe.unet.parameters(): p.requires_grad_(False)
    for p in pipe.vae.parameters(): p.requires_grad_(False)
    for p in pipe.text_encoder.parameters(): p.requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)

    bank = ScaleGateBank(
        q_max=args.q_max,
        q_prune=args.q_prune,
        shift_clip=args.shift_clip,
        hard_prune_forward=args.hard_prune_forward,
        train_shift=args.train_shift,
    ).to(device=device, dtype=torch.float32)
    n_hooks = install_gate(pipe, bank)

    images = load_images(args.image_dir, n=args.image_n, size=args.image_size, device=device)
    text_in = pipe.tokenizer(args.prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        text = pipe.text_encoder(text_in.input_ids.to(device))[0].to(dtype=dtype)
        latents_all = pipe.vae.encode(images.to(dtype=dtype)).latent_dist.sample(
            generator=torch.Generator(device=device).manual_seed(args.seed)
        ) * pipe.vae.config.scaling_factor

    print(f"[calib] hooks={n_hooks} running {args.calib_steps} calibration forwards...", flush=True)
    bank._calib_mode = True
    bank.enabled = False
    n_imgs = int(latents_all.shape[0])
    cal_rng = torch.Generator(device=device).manual_seed(args.seed + 7)
    with torch.no_grad():
        for _ in range(args.calib_steps):
            idx = torch.randint(0, n_imgs, (args.batch_n,), generator=cal_rng, device=device)
            text_idx = torch.randint(0, text.shape[0], (args.batch_n,), generator=cal_rng, device=device)
            noise = torch.randn(latents_all[idx].shape, generator=cal_rng, device=device, dtype=dtype)
            timesteps = torch.randint(0, int(train_sched.config.num_train_timesteps), (args.batch_n,), generator=cal_rng, device=device).long()
            noisy = train_sched.add_noise(latents_all[idx], noise, timesteps)
            _ = pipe.unet(noisy, timesteps, encoder_hidden_states=text[text_idx]).sample
    bank._calib_mode = False
    bank.enabled = True
    base_M = bank.base_M_from_calib()
    if not base_M:
        raise RuntimeError("no gate sites were calibrated; hook selection failed")
    all_M = torch.cat([v.reshape(-1).cpu() for v in base_M.values()])
    print(f"[calib] sites={len(base_M)} channels={all_M.numel()} min={all_M.min().item():.4g} mean={all_M.mean().item():.4g} max={all_M.max().item():.4g}", flush=True)

    m_q = {k: torch.zeros_like(v) for k, v in bank.params.items()}
    v_q = {k: torch.zeros_like(v) for k, v in bank.params.items()}
    m_b = {k: torch.zeros_like(v) for k, v in bank.shifts.items()}
    v_b = {k: torch.zeros_like(v) for k, v in bank.shifts.items()}

    rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    history = []
    n_cap_q = 0
    n_cap_b = 0
    n_nonfinite = 0
    t0 = time.time()
    log_every = args.log_every or max(1, args.steps // 30)

    for step in range(1, int(args.steps) + 1):
        idx = torch.randint(0, n_imgs, (args.batch_n,), generator=rng, device=device)
        text_idx = torch.randint(0, text.shape[0], (args.batch_n,), generator=rng, device=device)
        noise = torch.randn(latents_all[idx].shape, generator=rng, device=device, dtype=dtype)
        timesteps = torch.randint(0, int(train_sched.config.num_train_timesteps), (args.batch_n,), generator=rng, device=device).long()
        noisy = train_sched.add_noise(latents_all[idx], noise, timesteps)

        for p in bank.parameters():
            if p.grad is not None:
                p.grad = None
        pred = pipe.unet(noisy, timesteps, encoder_hidden_states=text[text_idx]).sample
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()

        bc1 = 1.0 - args.beta1 ** step
        bc2 = 1.0 - args.beta2 ** step
        update_q: dict[str, torch.Tensor] = {}
        update_b: dict[str, torch.Tensor] = {}

        for k, q in bank.params.items():
            g = q.grad if q.grad is not None else torch.zeros_like(q)
            m_q[k].mul_(args.beta1).add_(g, alpha=1.0 - args.beta1)
            v_q[k].mul_(args.beta2).addcmul_(g, g, value=1.0 - args.beta2)
            adam = (m_q[k] / bc1) / ((v_q[k] / bc2).sqrt() + args.eps)
            active, _M_actual, M_eff = _active_metric(base_M[k].to(q.device), q, metric_floor=args.metric_floor, q_prune=args.q_prune, m_damping=args.m_damping)
            u = adam / M_eff.sqrt()
            u = torch.where(active, u, torch.zeros_like(u))
            if args.max_delta_q > 0:
                u = _clip_signed_update(u, args.max_delta_q / max(float(args.lr), 1e-30))
            if not torch.isfinite(u).all():
                n_nonfinite += int((~torch.isfinite(u)).sum().item())
                u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            update_q[k] = u

        if args.train_shift:
            for k, b in bank.shifts.items():
                g = b.grad if b.grad is not None else torch.zeros_like(b)
                m_b[k].mul_(args.beta1).add_(g, alpha=1.0 - args.beta1)
                v_b[k].mul_(args.beta2).addcmul_(g, g, value=1.0 - args.beta2)
                u = (m_b[k] / bc1) / ((v_b[k] / bc2).sqrt() + args.eps)
                if args.max_delta_b > 0:
                    u = _clip_signed_update(u, args.max_delta_b / max(float(args.lr), 1e-30))
                if not torch.isfinite(u).all():
                    n_nonfinite += int((~torch.isfinite(u)).sum().item())
                    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                update_b[k] = u
        else:
            for k, b in bank.shifts.items():
                update_b[k] = torch.zeros_like(b)

        q_norm_sq = 0.0
        for k, u in update_q.items():
            q = bank.params[k]
            active, M_actual, _M_eff = _active_metric(base_M[k].to(q.device), q, metric_floor=args.metric_floor, q_prune=args.q_prune, m_damping=args.m_damping)
            q_norm_sq += float((M_actual[active] * u[active] * u[active]).sum().detach().cpu().item())
        q_norm = math.sqrt(max(0.0, q_norm_sq))
        if args.tau_q > 0 and math.isfinite(q_norm) and q_norm > args.tau_q:
            scale = float(args.tau_q) / q_norm
            for k in update_q:
                update_q[k] = update_q[k] * scale
            n_cap_q += 1

        b_norm_sq = 0.0
        for u in update_b.values():
            b_norm_sq += float((u * u).sum().detach().cpu().item())
        b_norm = math.sqrt(max(0.0, b_norm_sq))
        if args.tau_b > 0 and math.isfinite(b_norm) and b_norm > args.tau_b:
            scale = float(args.tau_b) / b_norm
            for k in update_b:
                update_b[k] = update_b[k] * scale
            n_cap_b += 1

        with torch.no_grad():
            for k, q in bank.params.items():
                q.add_(update_q[k], alpha=-float(args.lr))
                q.clamp_(min=float(args.q_prune), max=float(args.q_max))
            for k, b in bank.shifts.items():
                if args.train_shift:
                    b.add_(update_b[k], alpha=-float(args.lr))
                    b.clamp_(min=-float(args.shift_clip), max=float(args.shift_clip))
                else:
                    b.zero_()

        l = float(loss.detach().cpu().item())
        qsum = bank.q_summary()
        max_b = max((float(v.detach().abs().max().item()) for v in bank.shifts.values()), default=0.0)
        history.append({
            "step": step,
            "loss": l,
            "q_norm_active": q_norm,
            "b_norm": b_norm,
            "dead_frac": qsum["dead_frac"],
            "q_min": qsum["min"],
            "q_max": qsum["max"],
            "max_abs_b": max_b,
            "cap_q": n_cap_q,
            "cap_b": n_cap_b,
            "nonfinite_updates": n_nonfinite,
        })
        if step == 1 or step % log_every == 0 or step == args.steps:
            print(
                f"[step {step:>4}] loss={l:.4e} q=[{qsum['min']:.3f},{qsum['max']:.3f}] "
                f"dead={100*qsum['dead_frac']:.2f}% max|b|={max_b:.3f} "
                f"||u_q||_M(active)={q_norm:.3f} ||u_b||={b_norm:.3f} "
                f"caps(q/b)={n_cap_q}/{n_cap_b} nonfinite={n_nonfinite} ({time.time()-t0:.0f}s)",
                flush=True,
            )

    meta = {
        "steps": int(args.steps),
        "lr": float(args.lr),
        "tau_q": float(args.tau_q),
        "tau_b": float(args.tau_b),
        "metric_floor": float(args.metric_floor),
        "q_prune": float(args.q_prune),
        "q_max": float(args.q_max),
        "n_cap_q": int(n_cap_q),
        "n_cap_b": int(n_cap_b),
        "n_nonfinite": int(n_nonfinite),
        "history_last": history[-1] if history else {},
        "history_tail": history[-20:],
    }
    bank.save(out_path, meta=meta, base_M=base_M)
    (out_path.with_suffix(out_path.suffix + ".history.json")).write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[done] saved {out_path} cap_q={n_cap_q} cap_b={n_cap_b} nonfinite={n_nonfinite} total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

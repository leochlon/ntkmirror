from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import torch

from .data import Example, make_batch
from .losses import causal_loss_from_logits


@dataclass
class LogitProjectionSpec:
    """Fixed projection of supervised next-token logits for one local field solve."""

    mode: str
    mask: torch.Tensor
    indices: torch.Tensor | None = None
    top_k: int = 0
    center: bool = True
    vocab_size: int = 0
    n_positions: int = 0

    @property
    def field_dim(self) -> int:
        if self.mode == "full":
            return int(self.n_positions * self.vocab_size)
        if self.indices is None:
            return int(self.n_positions)
        return int(self.indices.numel())

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "top_k": int(self.top_k),
            "center": bool(self.center),
            "vocab_size": int(self.vocab_size),
            "n_positions": int(self.n_positions),
            "field_dim": int(self.field_dim),
        }


# Backwards-compatible public name exported from __init__.
LogitProjection = LogitProjectionSpec


def _supervised_shift(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [batch, seq, vocab] and labels [batch, seq]")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if not bool(mask.any()):
        raise ValueError("batch contains no supervised next-token labels")
    return shift_logits, shift_labels, mask


def build_logit_projection(
    baseline_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: str = "target",
    top_k: int = 32,
    center: bool = True,
    max_dim: int | None = None,
) -> LogitProjectionSpec:
    """Build a fixed projection from a baseline calibration forward pass.

    ``target``/``gold`` keeps one teacher-forced coordinate per supervised token.
    ``topk``/``gold_topk`` keeps the teacher token plus top-k baseline tokens.
    ``full`` keeps all vocabulary coordinates and is exact but often huge.
    """
    mode = str(mode).lower().strip()
    if mode == "gold":
        mode = "target"
    if mode == "gold_topk":
        mode = "topk"
    if mode not in {"target", "topk", "full"}:
        raise ValueError("projection mode must be one of: target/gold, topk/gold_topk, full")
    shift_logits, shift_labels, mask = _supervised_shift(baseline_logits, labels)
    selected_labels = shift_labels[mask].detach().long().cpu()
    n_positions = int(selected_labels.numel())
    vocab_size = int(shift_logits.shape[-1])
    if mode == "full":
        dim = n_positions * vocab_size
        if max_dim is not None and dim > int(max_dim):
            raise ValueError(f"full projection would create {dim} coordinates > max_dim={max_dim}")
        return LogitProjectionSpec(mode, mask.detach().cpu(), None, 0, bool(center), vocab_size, n_positions)
    if mode == "target":
        indices = selected_labels.view(-1, 1)
        dim = int(indices.numel())
    else:
        k = max(1, min(int(top_k), vocab_size))
        base_selected = shift_logits.detach()[mask].float().cpu()
        top = torch.topk(base_selected, k=k, dim=-1).indices.long()
        indices = torch.cat([selected_labels.view(-1, 1), top], dim=1)
        dim = int(indices.numel())
    if max_dim is not None and dim > int(max_dim):
        raise ValueError(f"projection would create {dim} coordinates > max_dim={max_dim}")
    return LogitProjectionSpec(mode, mask.detach().cpu(), indices.detach().cpu(), int(top_k), bool(center), vocab_size, n_positions)


def apply_logit_projection(logits: torch.Tensor, labels: torch.Tensor, spec: LogitProjectionSpec) -> torch.Tensor:
    shift_logits, _shift_labels, _mask_now = _supervised_shift(logits, labels)
    mask = spec.mask.to(device=logits.device)
    values = shift_logits[mask].float()
    if values.shape[0] != spec.n_positions:
        raise ValueError(f"projection expected {spec.n_positions} positions, got {values.shape[0]}")
    if spec.mode == "full":
        if spec.center:
            values = values - values.mean(dim=-1, keepdim=True)
        return values.reshape(-1).contiguous()
    if spec.indices is None:
        raise ValueError("non-full projection requires fixed indices")
    if spec.center:
        values_for_gather = values - values.mean(dim=-1, keepdim=True)
    else:
        values_for_gather = values
    gathered = values_for_gather.gather(1, spec.indices.to(device=logits.device))
    if spec.mode == "topk" and spec.center and gathered.shape[-1] > 1:
        # Remove the constant shift inside the scored event set as well.
        gathered = gathered - gathered.mean(dim=-1, keepdim=True)
    return gathered.reshape(-1).contiguous()


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float()).item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    den = torch.linalg.vector_norm(aa) * torch.linalg.vector_norm(bb)
    if float(den.item()) <= 1e-30:
        return 0.0
    return float((torch.dot(aa, bb) / den).item())


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.detach().float().reshape(-1)
    bb = b.detach().float().reshape(-1)
    den = max(float(torch.linalg.vector_norm(aa).item()), 1e-30)
    return float(torch.linalg.vector_norm(aa - bb).item() / den)


@contextlib.contextmanager
def _attached(controller):
    was_attached = bool(getattr(controller, "_handles", []))
    if not was_attached:
        controller.attach()
    try:
        yield
    finally:
        if not was_attached:
            controller.remove()


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _forward_logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"), use_cache=False)
    return out.logits


def projected_field(model, batch: dict[str, torch.Tensor], spec: LogitProjectionSpec, *, controller=None, signed_values: torch.Tensor | None = None) -> torch.Tensor:
    if controller is None:
        return apply_logit_projection(_forward_logits(model, batch), batch["labels"], spec)
    with _attached(controller):
        if signed_values is None:
            return apply_logit_projection(_forward_logits(model, batch), batch["labels"], spec)
        with controller.temporary_signed_log_values(signed_values):
            return apply_logit_projection(_forward_logits(model, batch), batch["labels"], spec)


@dataclass
class CGInfo:
    iterations: int
    residual_norm: float
    residual_history: list[float] = field(default_factory=list)
    converged: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def cg_solve(matvec: Callable[[torch.Tensor], torch.Tensor], b: torch.Tensor, *, max_iter: int = 32, tol: float = 1e-5) -> tuple[torch.Tensor, CGInfo]:
    x = torch.zeros_like(b)
    r = b.detach().float().clone()
    p = r.clone()
    rs_old = torch.dot(r, r)
    bnorm = max(float(torch.sqrt(rs_old).item()), 1e-30)
    hist = [float(torch.sqrt(rs_old).item())]
    converged = hist[-1] <= tol * bnorm
    it = 0
    for it in range(1, int(max_iter) + 1):
        if converged:
            break
        ap = matvec(p).detach().float()
        denom = torch.dot(p, ap)
        if not torch.isfinite(denom) or float(denom.item()) <= 1e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = torch.dot(r, r)
        resid = float(torch.sqrt(torch.clamp(rs_new, min=0.0)).item())
        hist.append(resid)
        if resid <= tol * bnorm:
            converged = True
            break
        p = r + (rs_new / rs_old.clamp_min(1e-30)) * p
        rs_old = rs_new
    return x, CGInfo(it, hist[-1], hist, converged)


@dataclass
class SecantDiagnostics:
    projection: dict
    alphas: list[dict]
    tangent_drift: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def controller_secant_diagnostics(
    model,
    tokenizer,
    controller,
    examples: Sequence[Example],
    *,
    alphas: Sequence[float] = (0.125, 0.25, 0.5, 0.75, 1.0),
    drift_alphas: Sequence[float] = (0.0625, 0.1875, 0.375, 0.625, 0.875),
    eps: float = 1e-3,
    batch_size: int = 2,
    max_length: int = 1024,
    projection: str = "target",
    top_k: int = 32,
) -> SecantDiagnostics:
    device = _device_of(model)
    batch = make_batch(tokenizer, list(examples[: max(1, int(batch_size))]), device=device, max_length=max_length)
    model.eval()
    s = controller.s.detach().float().to(device)
    zero = torch.zeros_like(s)
    with torch.no_grad(), _attached(controller), controller.temporary_signed_log_values(zero):
        base_logits = _forward_logits(model, batch)
        spec = build_logit_projection(base_logits, batch["labels"], mode=projection, top_k=top_k)
        base = apply_logit_projection(base_logits, batch["labels"], spec)
    with torch.no_grad():
        f_plus = projected_field(model, batch, spec, controller=controller, signed_values=eps * s)
        f_minus = projected_field(model, batch, spec, controller=controller, signed_values=-eps * s)
        tangent = (f_plus - f_minus) / (2.0 * eps)
    rows = []
    for a in alphas:
        with torch.no_grad():
            disp = projected_field(model, batch, spec, controller=controller, signed_values=float(a) * s) - base
            pred = float(a) * tangent
        rows.append({
            "alpha": float(a),
            "secant_err": _relerr(disp, pred),
            "cos_disp_tangent": _cos(disp, pred),
            "disp_norm": _norm(disp),
            "linear_pred_norm": _norm(pred),
        })
    drift = []
    for a in drift_alphas:
        lo = max(0.0, float(a) - eps)
        hi = float(a) + eps
        with torch.no_grad():
            local = (
                projected_field(model, batch, spec, controller=controller, signed_values=hi * s)
                - projected_field(model, batch, spec, controller=controller, signed_values=lo * s)
            ) / max(1e-30, hi - lo)
        drift.append({"alpha": float(a), "cos_local_initial": _cos(local, tangent), "local_norm": _norm(local)})
    return SecantDiagnostics(spec.to_dict(), rows, drift)


@dataclass
class FullWeightTarget:
    field: torch.Tensor
    grad_norm: float
    weight_step_norm: float
    parameter_count: int
    parameter_tensors: int
    step_size: float
    finite_difference_norm: float
    support_loss: float

    def summary(self) -> dict:
        return {
            "grad_norm": float(self.grad_norm),
            "weight_step_norm": float(self.weight_step_norm),
            "parameter_count": float(self.parameter_count),
            "parameter_tensors": float(self.parameter_tensors),
            "step_size": float(self.step_size),
            "finite_difference_norm": float(self.finite_difference_norm),
            "support_loss": float(self.support_loss),
        }


def full_weight_sgd_field_finite_difference(
    model,
    support_batch: dict[str, torch.Tensor],
    calibration_batch: dict[str, torch.Tensor],
    spec: LogitProjectionSpec,
    *,
    controller=None,
    step_size: float = 1e-5,
    param_name_substrings: Sequence[str] | None = None,
    max_parameters: int = 0,
) -> FullWeightTarget:
    names_and_params = list(model.named_parameters())
    if param_name_substrings:
        needles = tuple(str(x) for x in param_name_substrings)
        names_and_params = [(n, p) for n, p in names_and_params if any(k in n for k in needles)]
    param_count = int(sum(p.numel() for _, p in names_and_params if p.is_floating_point()))
    if max_parameters and param_count > int(max_parameters):
        raise RuntimeError(f"target would differentiate {param_count} parameters, exceeding max_target_parameters={max_parameters}")
    old_req = {id(p): bool(p.requires_grad) for _, p in model.named_parameters()}
    old_ctrl_req = None if controller is None else bool(controller.raw.requires_grad)
    touched: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    stepped = False
    try:
        for _, p in model.named_parameters():
            p.requires_grad_(False)
        for _, p in names_and_params:
            if p.is_floating_point():
                p.requires_grad_(True)
        if controller is not None:
            controller.raw.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            base = projected_field(model, calibration_batch, spec, controller=controller).detach().float()
        with _attached(controller) if controller is not None else contextlib.nullcontext():
            logits = _forward_logits(model, support_batch)
            loss = causal_loss_from_logits(logits, support_batch["labels"])
        loss.backward()
        grad_sq = 0.0
        tensors = 0
        with torch.no_grad():
            for _, p in names_and_params:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                touched.append((p, g))
                grad_sq += float(g.float().pow(2).sum().item())
                tensors += 1
                p.add_(g, alpha=-float(step_size))
            stepped = True
            after = projected_field(model, calibration_batch, spec, controller=controller).detach().float()
            for p, g in reversed(touched):
                p.add_(g, alpha=float(step_size))
            stepped = False
        field = after - base
        grad_norm = math.sqrt(max(0.0, grad_sq))
        return FullWeightTarget(field, grad_norm, abs(float(step_size)) * grad_norm, param_count, tensors, float(step_size), _norm(field), float(loss.detach().item()))
    finally:
        if stepped:
            with torch.no_grad():
                for p, g in reversed(touched):
                    p.add_(g, alpha=float(step_size))
        for _, p in model.named_parameters():
            p.requires_grad_(old_req[id(p)])
            p.grad = None
        if controller is not None and old_ctrl_req is not None:
            controller.raw.requires_grad_(old_ctrl_req)
            controller.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)


@dataclass
class DualProjectionDiagnostics:
    projection: dict
    target: dict
    cg: dict
    field_residual: float
    field_cosine: float
    range_residual: float
    range_cosine: float
    realized_residual: float
    realized_cosine: float
    target_norm: float
    projected_norm: float
    realized_norm: float
    update_norm: float
    update_metric_norm: float
    max_abs_update: float
    ridge: float
    fd_eps: float
    jvp_mode: str
    metric: str
    adjoint_error: float
    symmetry_error: float
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateProjectionSolution:
    update: torch.Tensor
    projected_field: torch.Tensor
    realized_field: torch.Tensor
    diagnostics: DualProjectionDiagnostics


def _metric_inverse_mul(v: torch.Tensor, metric_diag: torch.Tensor | None) -> torch.Tensor:
    vv = v.detach().float()
    if metric_diag is None:
        return vv
    md = metric_diag.detach().float().to(vv.device).reshape_as(vv).clamp_min(1e-30)
    return vv / md


def _metric_norm(v: torch.Tensor, metric_diag: torch.Tensor | None) -> float:
    vv = v.detach().float().reshape(-1)
    if metric_diag is None:
        return _norm(vv)
    md = metric_diag.detach().float().to(vv.device).reshape(-1).clamp_min(0.0)
    return float(torch.sqrt(torch.clamp((md * vv * vv).sum(), min=0.0)).item())


def _random_unit_like(ref: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    x = torch.randn(ref.numel(), generator=gen, device=ref.device, dtype=torch.float32)
    n = torch.linalg.vector_norm(x).clamp_min(1e-30)
    return x / n


def _relative_scalar_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    af = float(a.detach().float().item())
    bf = float(b.detach().float().item())
    return abs(af - bf) / max(abs(af) + abs(bf), 1e-30)


def gate_activation_metric_diag(
    model,
    batch: dict[str, torch.Tensor],
    controller,
    *,
    eps: float = 1e-6,
    include_current_scale: bool = True,
) -> torch.Tensor:
    """Approximate diagonal activation-control metric for selected gates.

    For a log-scale gate, the local activation displacement is
    ``delta h[..., c] = exp(s_c) * h[..., c] * delta s_c``. This returns
    ``E[h_c^2] * exp(2s_c)`` on the calibration batch. It is a diagonal
    pullback metric in activation space, not a full virtual-weight metric.
    """
    if controller is None:
        raise ValueError("controller is required for activation metric")
    device = _device_of(model)
    n = int(controller.layer_indices.numel())
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}
    layer_ids = sorted(set(int(x) for x in controller.layer_indices.detach().cpu().tolist()))
    layer_set = set(layer_ids)
    handles = []

    def record(layer_id: int, h: torch.Tensor) -> None:
        if not torch.is_tensor(h) or h.ndim < 2:
            return
        hf = h.detach().float()
        if hf.ndim == 3:
            ss = (hf * hf).mean(dim=(0, 1))
        else:
            ss = (hf.reshape(-1, hf.shape[-1]) ** 2).mean(dim=0)
        sums[layer_id] = sums.get(layer_id, torch.zeros_like(ss, device="cpu")) + ss.cpu()
        counts[layer_id] = counts.get(layer_id, 0) + 1

    def make_pre(layer_id: int):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                record(layer_id, inputs[0])
            return inputs
        return hook

    def make_post(layer_id: int):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            record(layer_id, h)
            return output
        return hook

    was_attached = bool(getattr(controller, "is_attached", False))
    if was_attached:
        controller.remove()
    try:
        for i, layer in enumerate(controller.layers):
            if i not in layer_set:
                continue
            if getattr(controller, "hook_site", "layer_output") == "layer_input":
                handles.append(layer.register_forward_pre_hook(make_pre(i)))
            else:
                handles.append(layer.register_forward_hook(make_post(i)))
        with torch.no_grad():
            _ = _forward_logits(model, batch)
    finally:
        for h in handles:
            h.remove()
        if was_attached:
            controller.attach()

    out = torch.empty(n, dtype=torch.float32, device=device)
    for i, (layer_id, channel_id) in enumerate(zip(
        controller.layer_indices.detach().cpu().tolist(),
        controller.channel_indices.detach().cpu().tolist(),
        strict=True,
    )):
        layer_id = int(layer_id)
        channel_id = int(channel_id)
        if layer_id not in sums or counts.get(layer_id, 0) <= 0:
            out[i] = float(eps)
        else:
            ss = sums[layer_id] / max(1, counts[layer_id])
            if channel_id >= ss.numel():
                out[i] = float(eps)
            else:
                out[i] = float(ss[channel_id].item())
    out = out.clamp_min(float(eps))
    if include_current_scale:
        s_now = controller.s.detach().float().to(device).reshape_as(out)
        out = out * torch.exp(2.0 * s_now).clamp_min(0.0)
    return out.clamp_min(float(eps))


def solve_gate_projection_matrix_free(
    model,
    calibration_batch: dict[str, torch.Tensor],
    spec: LogitProjectionSpec,
    controller,
    target_field: torch.Tensor,
    *,
    ridge: float = 1e-4,
    cg_iters: int = 16,
    cg_tol: float = 1e-5,
    fd_eps: float = 1e-3,
    metric_diag: torch.Tensor | None = None,
    metric_name: str = "identity",
    jvp_mode: str = "exact",
    diagnostic_seed: int = 0,
) -> GateProjectionSolution:
    """Project a full-weight target field into the gate tangent.

    The solved system is

        (B M^{-1} B^T + ridge I) alpha = target,
        update = M^{-1} B^T alpha.

    By default ``Bv`` is an exact autograd JVP and ``B^T y`` is an exact VJP,
    so the CG operator is symmetric PSD up to floating-point/autograd error. A
    finite-difference JVP is still available only as an explicit legacy mode.
    """
    t0 = time.perf_counter()
    model.eval()
    device = _device_of(model)
    target = target_field.detach().float().to(device).reshape(-1)
    s0 = controller.s.detach().float().to(device).reshape(-1)
    if metric_diag is not None:
        metric_diag = metric_diag.detach().float().to(device).reshape_as(s0).clamp_min(1e-30)
    jvp_mode = str(jvp_mode).lower().strip()
    if jvp_mode not in {"exact", "fd"}:
        raise ValueError("jvp_mode must be 'exact' or 'fd'")

    def field_at(signed_values: torch.Tensor, *, grad: bool) -> torch.Tensor:
        signed_values = signed_values.reshape_as(s0)
        if grad:
            with _attached(controller), controller.temporary_signed_log_values(signed_values):
                return apply_logit_projection(_forward_logits(model, calibration_batch), calibration_batch["labels"], spec).float().reshape(-1)
        with torch.no_grad(), _attached(controller), controller.temporary_signed_log_values(signed_values):
            return apply_logit_projection(_forward_logits(model, calibration_batch), calibration_batch["labels"], spec).float().reshape(-1)

    def b_v_exact(v: torch.Tensor) -> torch.Tensor:
        vv = v.detach().float().to(device).reshape_as(s0)
        def func(x: torch.Tensor) -> torch.Tensor:
            return field_at(x, grad=True)
        _y, jvp = torch.autograd.functional.jvp(func, (s0.detach().clone().requires_grad_(True),), (vv,), create_graph=False, strict=False)
        return jvp.detach().float().reshape(-1)

    def b_v_fd(v: torch.Tensor) -> torch.Tensor:
        vv = v.detach().float().to(device).reshape_as(s0)
        scale = max(1.0, float(torch.linalg.vector_norm(vv).item()))
        eps = float(fd_eps) / scale
        return (field_at(s0 + eps * vv, grad=False) - field_at(s0 - eps * vv, grad=False)) / (2.0 * eps)

    def b_v(v: torch.Tensor) -> torch.Tensor:
        return b_v_exact(v) if jvp_mode == "exact" else b_v_fd(v)

    def bt_y(y: torch.Tensor) -> torch.Tensor:
        yy = y.detach().float().to(device).reshape(-1)
        s_var = s0.detach().clone().requires_grad_(True)
        f = field_at(s_var, grad=True)
        (grad_s,) = torch.autograd.grad(torch.dot(f.float().reshape(-1), yy), s_var, retain_graph=False, create_graph=False)
        return grad_s.detach().float().reshape(-1)

    def k_matvec(y: torch.Tensor) -> torch.Tensor:
        yy = y.detach().float().to(device).reshape_as(target)
        return b_v(_metric_inverse_mul(bt_y(yy), metric_diag)) + float(ridge) * yy

    gen = torch.Generator(device=device).manual_seed(int(diagnostic_seed))
    v_probe = _random_unit_like(s0, gen)
    y_probe = _random_unit_like(target, gen)
    bv = b_v(v_probe)
    bty = bt_y(y_probe)
    adjoint_error = _relative_scalar_gap(torch.dot(bv, y_probe), torch.dot(v_probe, bty))
    a_probe = _random_unit_like(target, gen)
    b_probe = _random_unit_like(target, gen)
    Ka = k_matvec(a_probe)
    Kb = k_matvec(b_probe)
    symmetry_error = _relative_scalar_gap(torch.dot(a_probe, Kb), torch.dot(b_probe, Ka))

    alpha, cg_info = cg_solve(k_matvec, target, max_iter=cg_iters, tol=cg_tol)
    update = _metric_inverse_mul(bt_y(alpha), metric_diag)
    projected = b_v(update).detach().float().reshape(-1)
    with torch.no_grad():
        base = field_at(s0, grad=False).detach().float().reshape(-1)
        realized = (field_at(s0 + update, grad=False).detach().float().reshape(-1) - base)
    range_residual = _relerr(target, projected)
    realized_residual = _relerr(target, realized)
    diag = DualProjectionDiagnostics(
        projection=spec.to_dict(),
        target={"mode": "full_weight_sgd_finite_difference", "field_dim": int(target.numel())},
        cg=cg_info.to_dict(),
        field_residual=realized_residual,
        field_cosine=_cos(target, realized),
        range_residual=range_residual,
        range_cosine=_cos(target, projected),
        realized_residual=realized_residual,
        realized_cosine=_cos(target, realized),
        target_norm=_norm(target),
        projected_norm=_norm(projected),
        realized_norm=_norm(realized),
        update_norm=_norm(update),
        update_metric_norm=_metric_norm(update, metric_diag),
        max_abs_update=float(update.detach().abs().max().item()) if update.numel() else 0.0,
        ridge=float(ridge),
        fd_eps=float(fd_eps),
        jvp_mode=jvp_mode,
        metric=str(metric_name),
        adjoint_error=float(adjoint_error),
        symmetry_error=float(symmetry_error),
        elapsed_seconds=float(time.perf_counter() - t0),
    )
    return GateProjectionSolution(
        update=update.detach().cpu(),
        projected_field=projected.detach().cpu(),
        realized_field=realized.detach().cpu(),
        diagnostics=diag,
    )


def solve_controller_field_update(model, controller, batch, projection, target_field, **kwargs):
    return solve_gate_projection_matrix_free(model, batch, projection, controller, target_field, **kwargs)

def dual_projection_diagnostics(
    model,
    tokenizer,
    controller,
    support_examples: Sequence[Example],
    calibration_examples: Sequence[Example] | None = None,
    *,
    batch_size: int = 1,
    max_length: int = 1024,
    projection: str = "target",
    top_k: int = 32,
    target_step_size: float = 1e-5,
    ridge: float = 1e-4,
    cg_iters: int = 16,
    cg_tol: float = 1e-5,
    fd_eps: float = 1e-3,
    metric: str = "identity",
    metric_eps: float = 1e-6,
    jvp_mode: str = "exact",
    param_name_substrings: Sequence[str] | None = None,
    max_target_parameters: int = 0,
) -> tuple[GateProjectionSolution, FullWeightTarget]:
    device = _device_of(model)
    support = list(support_examples[: max(1, int(batch_size))])
    calibration_source = calibration_examples if calibration_examples is not None else support_examples
    calibration = list(calibration_source[: max(1, int(batch_size))])
    support_batch = make_batch(tokenizer, support, device=device, max_length=max_length)
    calibration_batch = make_batch(tokenizer, calibration, device=device, max_length=max_length)
    model.eval()
    with torch.no_grad(), _attached(controller):
        base_logits = _forward_logits(model, calibration_batch)
        spec = build_logit_projection(base_logits, calibration_batch["labels"], mode=projection, top_k=top_k)
    target = full_weight_sgd_field_finite_difference(
        model,
        support_batch,
        calibration_batch,
        spec,
        controller=controller,
        step_size=target_step_size,
        param_name_substrings=param_name_substrings,
        max_parameters=max_target_parameters,
    )
    metric_name = str(metric).lower().strip()
    if metric_name == "identity":
        metric_diag = None
    elif metric_name == "activation":
        metric_diag = gate_activation_metric_diag(
            model, calibration_batch, controller, eps=metric_eps, include_current_scale=True
        )
    else:
        raise ValueError("metric must be 'identity' or 'activation'")
    solution = solve_gate_projection_matrix_free(
        model,
        calibration_batch,
        spec,
        controller,
        target.field,
        ridge=ridge,
        cg_iters=cg_iters,
        cg_tol=cg_tol,
        fd_eps=fd_eps,
        metric_diag=metric_diag,
        metric_name=metric_name,
        jvp_mode=jvp_mode,
    )
    solution.diagnostics.target.update(target.summary())
    return solution, target

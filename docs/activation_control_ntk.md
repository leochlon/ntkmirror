# Activation-control NTK duality

The controller is not certified by being small in its own gate chart. A finite
gate vector can leave the initial secant approximation

```text
P z(s) - P z(0) ~= B(0) s
```

while still being useful. The theory-facing object is the projected output
field. At a current controller state `s_t`, the selected scalar gates have local
activation-control Jacobian

```text
B_C(s_t) = d(P_C z(s_t)) / ds .
```

The full frozen-model weight NTK field for a support loss gradient is

```text
d_C^theta = -eta J_{theta,C} J_{theta,S}^T g_S .
```

A scalar gate update is locally NTK-dual on the scored projection only when
`d_C^theta` lies in the range of `B_C(s_t)`, or when the residual after projecting
onto that range is small. This replaces gate-norm, clip-fraction, and
single-initial-tangent certificates.

## Adjoint-consistent solver

CG is only valid for a symmetric positive-definite operator. `B v` is an exact
autograd JVP and `B^T y` an exact autograd VJP, so the solver is
adjoint-consistent:

```text
B v      = exact autograd JVP
B^T y    = exact autograd VJP
K y      = B M^{-1} B^T y + lambda y
u        = M^{-1} B^T alpha
```

A finite-difference JVP is available as `--jvp-mode fd` for ablation only;
mixing it with the autograd VJP makes `K` non-adjoint, so its residuals are not
reachability evidence. The default is `--jvp-mode exact`.

The solver reports separate residuals:

- `range_residual`: local linear range residual, `||d - B u|| / ||d||`.
- `realized_residual`: actual forward residual, `||d - (F(s+u)-F(s))|| / ||d||`.
- `field_residual`: alias for `realized_residual`.
- `adjoint_error`: random-probe check of `<Bv,y> = <v,B^T y>`.
- `symmetry_error`: random-probe check of `a^T K b = b^T K a`.
- `clip_fraction` and `clip_update_residual` in `fit-dual`: whether the solved
  signed-log update was changed by `max_log_gate` clipping.

## Metrics

The field solve is

```text
alpha = (B M^{-1} B^T + lambda I)^{-1} d,
u     = M^{-1} B^T alpha.
```

`--metric identity` keeps the Euclidean gate metric. It is an ablation, not a
weight-norm-correct claim. `--metric activation` uses a diagonal activation
metric for the selected gates:

```text
M_j ~= E[h_j^2] exp(2 s_j)
```

on the calibration batch. This is a practical activation-control metric, not a
full virtual-weight metric.

## Commands

Secant diagnostic for a fitted controller:

```bash
ntkmirror secant \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --controller runs/controller.pt \
  --eval eval.jsonl \
  --projection topk --top-k 32
```

Dual range diagnostic:

```bash
ntkmirror dual-diagnose \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --support train.jsonl \
  --calibration eval.jsonl \
  --gates 57000 \
  --hook-site layer_input \
  --projection topk --top-k 32 \
  --target-step-size 1e-5 \
  --jvp-mode exact \
  --metric activation \
  --cg-iters 16 \
  --ridge 1e-4 \
  --out dual_report.json
```

Pathwise field-locked fit:

```bash
ntkmirror fit-dual \
  --model Qwen/Qwen2.5-1.5B \
  --train train.jsonl \
  --out controller_dual.pt \
  --gates 57000 \
  --hook-site layer_input \
  --projection topk --top-k 32 \
  --steps 8 \
  --target-step-size 1e-5 \
  --jvp-mode exact \
  --metric activation \
  --cg-iters 16
```

Use `--projection full` only when the supervised-token count and vocabulary size
fit memory. `target` is cheap but only tests the teacher-token event coordinate;
`topk` is the practical default for code and math probes.

## Diffusion scale-gate script

The experimental diffusion runner is under:

```text
scripts/diffusion/train_scale_gate_adam_m.py
```

It keeps activation-energy calibration and Adam + `M_t = base_M exp(2q)`, but it
removes the unsafe `-inf`/NaN leak. Channels can still self-prune, but they do so
through a finite `q_prune` floor and a hard-dead mask. Dead channels are excluded
from the active q-update M-norm cap, so they cannot become a global learning-rate
divisor.

## Failure modes to check

1. **Range failure.** If `range_residual` is high and CG/adjoint/symmetry checks
   are clean, the full-weight SGD field is not in the current selected gate
   tangent on the chosen projection.
2. **Realization failure.** If `range_residual` is low but `realized_residual` is
   high, the local step is too large or the controller chart is too curved.
3. **Projection too narrow.** `target` may look good while `topk` or `full` fails.
4. **Target finite-difference scale.** Too large `--target-step-size` measures a
   nonlinear weight move; too small can drown in numerical noise. Sweep it.
5. **CG under-solve.** A high residual with poor CG convergence may be a solver
   failure. Increase `--cg-iters` and inspect `adjoint_error`/`symmetry_error`.
6. **FD/VJP mismatch.** `--jvp-mode fd` is an ablation fallback only. If its
   adjoint or symmetry errors are high, do not use its residuals as reachability
   evidence.
7. **Initial-tangent failure is not dual failure.** `secant_err` at alpha 1 can be
   high while pathwise local field residual is low.
8. **Full-weight target cost.** `dual-diagnose` computes gradients with respect
   to model weights. On 7B models this is expensive. Use small batches and
   `topk` first, and optionally `--param-filter` for ablations.
9. **Hook-site mismatch.** `layer_output` preserves existing controllers; for
   theorem-facing activation-input claims, run `layer_input` ablations.
10. **Controller clipping.** If `clip_fraction` is high, the applied step is not
    the solved step.

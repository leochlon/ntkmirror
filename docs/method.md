# Method note

`ntkmirror` implements the deployable controller, not the full research harness.
The base model is frozen. A sparse set of decoder-layer output channels is
selected by the magnitude of the log-gate derivative

```text
dL/ds_{l,c} = sum_t <dL/dh_{l,t,c}, h_{l,t,c}> .
```

The fitted controller is a shared signed log-mask

```text
h'_{l,t,c} = exp(s_{l,c}) h_{l,t,c},   |s_{l,c}| <= max_log_gate.
```

The same `s` is attached during support scoring, held-out evaluation, and
generation. No LoRA modules or permanent weight edits are inserted.

## Why this interface is small

The public API exposes one class and one CLI:

```python
from ntkmirror import ForwardFineTuner, load_jsonl_examples

tuner = ForwardFineTuner(model, tokenizer, gates=5000)
tuner.fit(load_jsonl_examples("train.jsonl"))
tuner.save("controller.pt")
```

The CLI is the same:

```bash
ntkmirror fit --model Qwen/Qwen2.5-0.5B-Instruct --train train.jsonl --out controller.pt
ntkmirror eval --model Qwen/Qwen2.5-0.5B-Instruct --controller controller.pt --eval eval.jsonl
```

## Failure modes to check before believing a result

1. **Base example selection.** If evaluation contains problems the base model
   already solves, exact-accuracy deltas can be washed out. Report base metrics.

2. **Controller capacity.** A 512-gate controller is a different object from a
   5k or 10k controller. Sweep `--gates` before making a LoRA comparison.

3. **NLL vs exact answer.** The controller can improve NLL without improving
   long-horizon exact-answer accuracy. Report both where possible.

4. **Gate budget.** Very large `--max-log-gate` can hurt generation. Do not use
   gate norm or clip fraction as the NTK-dual certificate; use `secant` for the
   initial-tangent check and `dual-diagnose` for the full-weight field-range
   check.

5. **Layer hooks.** This package gates decoder-layer outputs. Architectures
   without a standard decoder stack may need a layer-path extension.

6. **Seed manifests.** For benchmark claims, fix train/eval splits and compare
   every arm on the same examples.

7. **Oracle diagnostics.** Fitting a controller to an SGD displacement is useful
   for theorem diagnostics, but it is not a deployable fine-tuning method. The
   deployable path here is support-loss optimisation of the gates.

## Activation-control NTK dual diagnostics

`ntkmirror fit` performs deployable support-loss optimisation over signed log
gates. A separate research path treats the gates as an activation-control basis

```text
B_C(s) = d(P_C z(s)) / ds
```

and takes the target field to be the full model's empirical weight-SGD/NTK
velocity

```text
d_C^theta = -eta J_{theta,C} J_{theta,S}^T g_S .
```

The controller is dual only when this target field lies in the output range of
`B_C(s)` on the chosen projected-logit coordinates. The relevant residual is
therefore

```text
||d_C^theta - Pi_{B_C} d_C^theta|| / ||d_C^theta||,
```

not the raw gate norm and not global secant linearity from `s=0`. The old
local-controller statement `Pz(s)-Pz(0) ~= B(0)s` is only a secant diagnostic;
`fit` remains the cheap deployable support-loss optimiser and should not be
described as a proof of full-weight SGD duality without the range diagnostic.

### Commands

```bash
ntkmirror secant \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --controller controller.pt \
  --eval eval.jsonl \
  --projection topk --top-k 32
```

This diagnoses whether a finite trained controller follows its initial gate
chart tangent. A high secant error means the initial controller linearisation is
not a valid global explanation; it does not by itself prove the controller left
the full-weight NTK field geometry.

```bash
ntkmirror dual-diagnose \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --support train.jsonl \
  --controller controller.pt \
  --projection topk --top-k 32 \
  --target-step-size 1e-5 \
  --ridge 1e-4
```

This computes a tiny reversible full-weight SGD step, measures its projected
logit displacement, and projects that field onto the current gate tangent using
matrix-free products with `B` and `B^T`. The output field residual is the direct
range test for the duality claim.

```bash
ntkmirror fit-dual \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train train.jsonl \
  --out controller.dual.pt \
  --steps 8 \
  --projection topk --top-k 32 \
  --target-step-size 1e-5
```

This is the pathwise field-locked training mode. Each step recomputes the local
full-weight target field and applies the signed-log gate update that best
realises it in the current activation-control tangent.

### Hook sites

`--hook-site layer_output` preserves the original package behaviour;
`--hook-site layer_input` gates the input to each decoder layer instead. The
claim is not that the gate vector is small. It is that the activation-control
field induced at the selected hook sites can realise the full-weight SGD
projected-logit field, up to the measured range residual.

### Practical limits

The full-weight target stores ordinary parameter gradients for the selected
model parameters. On large models this is expensive. Use `--param-filter` for
scoped experiments, for example `--param-filter q_proj,k_proj,v_proj,o_proj`,
or set `--max-target-parameters` as a guard before intentionally differentiating
the whole model.

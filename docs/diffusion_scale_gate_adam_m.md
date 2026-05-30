# Diffusion Adam-M scale gate audit

The uploaded `train_scale_gate_adam_M.py` result is useful but the winning arm
must not be copied literally. The script calibrated per-channel activation
energy, used `M_t = base_M * exp(2q)`, and applied a post-Adam `1/sqrt(M_t)`
preconditioner. Those are worth keeping. The observed `q -> -inf` behaviour is
also a real clue: multiplicative gates can self-prune unused channels.

The unsafe part was that `q -> -inf` produced zero or non-finite `M_t`. That can
turn the global cap into a NaN-bypassed no-op. The clamp experiment then failed
for the opposite reason: clamped dead channels stayed finite, contributed a huge
number of small-metric coordinates to the global cap, and converted the cap into
a global learning-rate divider.

The replacement script is:

```text
scripts/diffusion/train_scale_gate_adam_m.py
```

Changes:

- no `inf` or `nan` updates are permitted;
- finite `q_prune` represents dead channels;
- hard-dead channels can have forward scale exactly zero;
- dead channels are excluded from the active q-update M-norm cap;
- q and shift updates have separate caps;
- per-coordinate `max_delta_q` and `max_delta_b` bound one-step damage;
- `M_eff = max(M_actual, metric_floor)` prevents division by zero;
- checkpoints store `base_M`, q/dead statistics, cap counts, and history tail.

The default `--tau-q 0 --tau-b 0` disables global caps because the best uploaded
run effectively bypassed its global cap after the `-inf` leak. Use per-coordinate
caps first; enable `--tau-q` only after inspecting active-channel norm logs.

`M_t = base_M * exp(2q)` is asymmetric. For `q -> -inf`, `M_t -> 0` and the
post-Adam step `u/sqrt(M_t) -> inf`. The optimiser drives negative drift
instead of damping it. Trained runs push 3-7% of channels to `q ~ -40` and
the gate soft-prunes prior content. `q_max` does not fix this; the lower
tail is unbounded.

Replace `exp(2q)` with `cosh(2q)`. `cosh` damps `|q| -> inf` symmetrically,
so the optimiser cannot drive channels to extreme attenuation.

```
                      q range           std    |q|>5
original cosh tau=500 [-5.06, +3.75]    0.78    0.0%
this script (cosh)    [-4.70, +3.57]    0.78    0.0%
this script (exp)     [-42, +2 to +3]   5.40   19.7%
```

The script defaults to `--m-damping cosh`. Use `--m-damping exp` to
reproduce the asymmetric exp-damping behaviour.

Failure modes to watch:

1. `dead_frac` goes to 100%: metric floor or LR is too aggressive, or the prompt/image loss is pushing global deletion.
2. `dead_frac` stays near 0%: pruning floor too low, LR too low, or Adam moments never produce decisive signs.
3. `caps(q/b)` increments every step: the cap is again a global LR divider.
4. `max|b|` dominates: shifts are learning additive biases, not scale-gate behaviour; run `--no-train-shift` as an ablation.
5. DINO improves but loss explodes: the gate found a feature-specific shortcut; evaluate prompt diversity and reconstruction quality.
6. DINO improves only with hard pruning: report it as structured channel pruning, not as an NTK-linear dual result.
7. DINO improves as `alpha` drops below 1.0 at inference: the trainer is using `--m-damping exp` and the lower q-tail ran to `~ -40`. Switch to `--m-damping cosh` so `alpha=1.0` is the deployable operating point.

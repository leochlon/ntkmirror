# NTK-Mirror

**Hassana Labs** — Leon Chlon ([lc574@cantab.ac.uk](mailto:lc574@cantab.ac.uk))

LoRA-free forward-pass fine-tuning for Hugging Face causal language models.

`ntkmirror` learns a small signed controller on top of a frozen Transformer. It
adds no LoRA modules and makes no permanent weight edits. The controller is a
sparse set of shared log-gates on decoder-layer output channels:

```text
h'_{layer, token, channel} = exp(s_{layer, channel}) h_{layer, token, channel}
```

The gates are learned from teacher-forced examples and then attached to the same
Hugging Face model during evaluation or generation.

## Install

```bash
git clone https://github.com/leochlon/ntkmirror.git
cd ntkmirror
pip install -e .
```

## Minimal use

Create `train.jsonl`:

```jsonl
{"prompt":"Question: 14 + 27 = ?\nAnswer:","completion":" 41"}
{"prompt":"Question: 36 + 18 = ?\nAnswer:","completion":" 54"}
```

Fit a controller:

```bash
ntkmirror fit \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train train.jsonl \
  --out controller.pt
```

Evaluate it:

```bash
ntkmirror eval \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --controller controller.pt \
  --eval eval.jsonl
```

Generate with it:

```bash
ntkmirror generate \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --controller controller.pt \
  --prompt "Question: 47 + 36 = ?\nAnswer:"
```

## One-command demo

```bash
pip install -e .
bash examples/run_demo.sh
```

For a smaller run:

```bash
GATES=512 STEPS=40 bash examples/run_demo.sh
```

## Python API

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ntkmirror import ForwardFineTuner, load_jsonl_examples

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").cuda()

tuner = ForwardFineTuner(model, tokenizer, gates=5000)
tuner.fit(load_jsonl_examples("train.jsonl"), steps=240)
tuner.save("controller.pt")

print(tuner.generate("Question: 47 + 36 = ?\nAnswer:"))
```

## Data format

Preferred JSONL schema:

```jsonl
{"prompt":"...context...","completion":"...teacher-forced target..."}
```

Also accepted:

```jsonl
{"instruction":"...","response":"..."}
{"question":"...","answer":"..."}
{"text":"..."}
```

## Important defaults

| Option | Default | Meaning |
|---|---:|---|
| `--gates` | `5000` | number of layer-channel log-gates |
| `--steps` | `240` | AdamW steps on gate parameters only |
| `--lr` | `5e-3` | controller learning rate |
| `--max-log-gate` | `0.05` | bound on each signed log-gate |
| `--layers` | `all` | decoder layers to score and gate |
| `--score-batches` | `16` | batches used to select gates |

## Compose two task controllers

Controllers are saved in signed log-gate coordinates, so composition is simple:
add the signed log-gates, clip to a safe budget, and attach the resulting
controller. This is the activation-space analogue of adding task directions,
except the addition happens in log-mask/mirror coordinates rather than LoRA
weight space.

```bash
ntkmirror compose \
  --controllers runs/gsm8k_controller.pt runs/mbpp_controller.pt \
  --out runs/gsm8k_plus_mbpp.pt \
  --report runs/composition_report.json

ntkmirror inspect \
  --controllers runs/gsm8k_controller.pt runs/mbpp_controller.pt runs/gsm8k_plus_mbpp.pt
```

A disjoint-task runner is included:

```bash
pip install -e '.[datasets]'
bash scripts/run_disjoint_composition.sh
```

It builds GSM8K and MBPP JSONL subsets, fits one controller per task, composes
them, and evaluates base / task-A / task-B / composed controllers on both eval
sets. See `docs/composability.md`.


## Persistent controller memory

A memory item can be stored as a controller: one controller per conversation,
document, user preference, task style, or procedure. At inference time,
`ntkmirror` retrieves relevant items, composes their signed log-gates, and
attaches the composed controller before generation. This injects retrieved
context through the forward pass without appending the memory text to the prompt.

Fit-and-store a memory controller:

```bash
ntkmirror memory add \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --store runs/memory \
  --id arithmetic-carrying \
  --train examples/math_train.jsonl \
  --text "worked addition arithmetic with carrying" \
  --tags math,arithmetic
```

Or register an existing controller:

```bash
ntkmirror memory add \
  --store runs/memory \
  --id arithmetic-carrying \
  --controller runs/arithmetic.pt \
  --text "two-digit addition with carrying: add ones, carry, then tens"
```

Retrieve, compose, and generate:

```bash
ntkmirror memory search \
  --store runs/memory \
  --query "solve an addition problem with carrying"

ntkmirror memory generate \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --store runs/memory \
  --query "addition with carrying" \
  --prompt "Problem: 47 + 36 = ?\nSolution:"
```

Try the demo:

```bash
bash examples/run_memory_demo.sh
```

The default retriever is a dependency-free lexical TF-IDF scorer. That is
intentional for first-run UX: the main bottleneck in controller memory is
retrieval quality, not controller storage. For production, replace the retriever
with an embedding or hybrid vector-store layer and keep the same `compose_states`
interface. See `docs/persistent_memory.md`.


## Activation-control NTK tools

The `fit` command trains signed log-gates by support NLL and remains the
deployable path. A separate research path adds diagnostics and a field-locked
fitting harness for the stricter NTK-dual claim: the local activation-control
tangent

```text
B_C(s) = d(P_C z(s)) / ds
```

should realise the full frozen-model weight-SGD projected-logit field

```text
d_C^theta = -eta J_{theta,C} J_{theta,S}^T g_S .
```

`Bv` is an exact autograd JVP, `B^T y` is an exact VJP, and the CG operator is
`B M^{-1} B^T + ridge I`. Reports include `adjoint_error`, `symmetry_error`,
`range_residual`, and the actual forward `realized_residual`; `field_residual`
is the realised-forward residual, not the same local matvec used inside the
solve.

Audit whether a selected gate basis can realise the full-weight field:

```bash
ntkmirror dual-diagnose \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --support train.jsonl \
  --calibration eval.jsonl \
  --controller controller.pt \
  --projection topk --top-k 32 \
  --target-step-size 1e-5 \
  --jvp-mode exact \
  --metric activation
```

Fit pathwise by matching the full-weight NTK field instead of using support-Adam:

```bash
ntkmirror fit-dual \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train train.jsonl \
  --out controller_dual.pt \
  --steps 8 \
  --projection topk --top-k 32 \
  --jvp-mode exact \
  --metric activation
```

Check whether a finite controller has left the initial gate tangent:

```bash
ntkmirror secant \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --controller controller.pt \
  --eval eval.jsonl
```

The important numbers are `range_residual` and `realized_residual`, not raw
gate norm. A large `secant` error only says the initial gate chart is no longer
a global linear model; it does not by itself refute pathwise NTK duality. See
`docs/activation_control_ntk.md` for the theory, command details, and failure
mode checklist.

A safe diffusion scale-gate runner is also included:

```bash
python scripts/diffusion/train_scale_gate_adam_m.py \
  --image-dir images \
  --prompts "a photo of sks dog" \
  --out runs/diffusion_scale_gates.pt \
  --steps 1500
```

It uses Adam with a step-adaptive activation metric and `cosh` self-damping, and
represents channel pruning with finite `q_prune` hard-dead masks, separate
q/shift caps, and non-finite guards.

## What this repo is not

The default UX remains the simple deployable support-Adam package. The
diagnostic and field-locked commands expose a research harness for NTK-vector
diagnostics and field-locked local updates; they are slower than `fit` and are
not the default first-run path.

## Notes for benchmark claims

Always report the base model, controller, and LoRA on the same train/eval
manifest. For exact-answer tasks, report exact accuracy and teacher-forced NLL.
For system claims, report adaptation time and peak memory. See
`docs/method.md` for failure modes.

## Citation

```bibtex
@software{chlon2026ntkmirror,
  author       = {Leon Chlon},
  title        = {{NTK-Mirror: LoRA-free forward-pass fine-tuning via signed log-mask controllers}},
  year         = {2026},
  organization = {Hassana Labs},
  url          = {https://github.com/leochlon/ntkmirror}
}
```

## License

MIT © 2026 Hassana Labs — Leon Chlon.

import torch
import torch.nn as nn

from ntkmirror.controller import _SignedLogMaskModule, SignedLogMaskState
from ntkmirror.dual import build_logit_projection, apply_logit_projection, cg_solve


def test_logit_projection_target_and_full_shapes():
    logits = torch.randn(2, 4, 7)
    labels = torch.tensor([[-100, 1, 2, -100], [-100, -100, 3, 4]])
    target = build_logit_projection(logits, labels, mode="target")
    full = build_logit_projection(logits, labels, mode="full")
    assert target.field_dim == 4
    assert full.field_dim == 4 * 7
    assert apply_logit_projection(logits, labels, target).numel() == target.field_dim
    assert apply_logit_projection(logits, labels, full).numel() == full.field_dim


def test_cg_solve_spd_system():
    a = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    x, info = cg_solve(lambda v: a @ v, b, max_iter=8, tol=1e-8)
    assert info.residual_norm < 1e-5
    assert torch.allclose(x, torch.linalg.solve(a, b), atol=1e-5)


def test_signed_log_override_and_state_backward_compatibility():
    layers = nn.ModuleList([nn.Identity()])
    module = _SignedLogMaskModule(
        layers,
        torch.tensor([0]),
        torch.tensor([1]),
        hidden_size=3,
        max_log_gate=0.1,
        hook_site="layer_output",
    )
    x = torch.ones(1, 1, 3)
    module.attach()
    try:
        with module.temporary_signed_log_values(torch.tensor([0.05])):
            y = layers[0](x)
        assert abs(float(y[0, 0, 1]) - float(torch.exp(torch.tensor(0.05)))) < 1e-6
    finally:
        module.remove()

    old_payload = {
        "layer_path": "model.layers",
        "n_layers": 1,
        "hidden_size": 3,
        "layer_indices": torch.tensor([0]),
        "channel_indices": torch.tensor([1]),
        "raw": torch.tensor([0.0]),
        "max_log_gate": 0.1,
        "model_name": "tiny",
    }
    state = SignedLogMaskState.from_dict(old_payload)
    assert state.hook_site == "layer_output"
    assert state.theory_version == "activation_control"


class _ToyOut:
    def __init__(self, logits):
        self.logits = logits


class _ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(11, 4)
        self.layers = nn.ModuleList([nn.Identity()])
        self.head = nn.Linear(4, 11, bias=False)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        h = self.emb(input_ids)
        h = self.layers[0](h)
        return _ToyOut(self.head(h))


def test_exact_jvp_vjp_projection_solver_has_small_adjoint_and_symmetry_error():
    from ntkmirror.dual import solve_gate_projection_matrix_free

    torch.manual_seed(0)
    model = _ToyLM()
    controller = _SignedLogMaskModule(
        model.layers,
        torch.tensor([0, 0]),
        torch.tensor([0, 2]),
        hidden_size=4,
        max_log_gate=1.0,
        hook_site="layer_output",
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3, 4], [-100, 3, 4, 5]], dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    controller.attach()
    try:
        base_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
        spec = build_logit_projection(base_logits, batch["labels"], mode="target")
        s0 = controller.s.detach().float()
        true_update = torch.tensor([1e-3, -7e-4])

        def field(x):
            with controller.temporary_signed_log_values(x):
                return apply_logit_projection(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits, batch["labels"], spec)

        _, target = torch.autograd.functional.jvp(
            field,
            (s0.detach().clone().requires_grad_(True),),
            (true_update,),
            create_graph=False,
            strict=False,
        )
    finally:
        controller.remove()

    sol = solve_gate_projection_matrix_free(
        model,
        batch,
        spec,
        controller,
        target.detach(),
        ridge=1e-10,
        cg_iters=32,
        cg_tol=1e-9,
        jvp_mode="exact",
    )
    d = sol.diagnostics
    assert d.adjoint_error < 1e-5
    assert d.symmetry_error < 1e-5
    assert d.range_residual < 1e-4
    assert d.realized_residual < 5e-3


def test_metric_projection_uses_metric_inverse_solution_shape():
    from ntkmirror.dual import solve_gate_projection_matrix_free

    torch.manual_seed(1)
    model = _ToyLM()
    controller = _SignedLogMaskModule(
        model.layers,
        torch.tensor([0, 0]),
        torch.tensor([1, 3]),
        hidden_size=4,
        max_log_gate=1.0,
        hook_site="layer_output",
    )
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2, 3]], dtype=torch.long),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    controller.attach()
    try:
        spec = build_logit_projection(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits, batch["labels"], mode="target")
        s0 = controller.s.detach().float()
        direction = torch.tensor([5e-4, 3e-4])
        def field(x):
            with controller.temporary_signed_log_values(x):
                return apply_logit_projection(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits, batch["labels"], spec)
        _, target = torch.autograd.functional.jvp(field, (s0.detach().clone().requires_grad_(True),), (direction,), strict=False)
    finally:
        controller.remove()
    metric = torch.tensor([2.0, 5.0])
    sol = solve_gate_projection_matrix_free(
        model, batch, spec, controller, target.detach(), ridge=1e-10, cg_iters=32,
        metric_diag=metric, metric_name="test_metric", jvp_mode="exact",
    )
    assert sol.update.numel() == 2
    assert sol.diagnostics.metric == "test_metric"
    assert sol.diagnostics.update_metric_norm > 0
    assert sol.diagnostics.range_residual < 1e-4

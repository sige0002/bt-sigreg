"""Independent CPU-only algebra/limitation checks for research A/B/C.

This is not LeWM integration, training, or evidence of control performance.
Run: .venv/bin/python mylewm/abc_math_checks.py
"""

import json

import torch


torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.manual_seed(20260907)
RESULTS = {}


def check(name, actual, expected, atol=1e-9):
    a, e = torch.as_tensor(actual), torch.as_tensor(expected)
    error = (a - e).abs().max().item()
    assert torch.allclose(a, e, atol=atol, rtol=1e-7), (name, a, e)
    RESULTS[name] = {"passed": True, "max_abs_error": error}


def covariance(m):
    x = m - m.mean(0)
    return x.T @ x / len(m)


def participation(c, eps=1e-8):
    return c.trace().square() / (c.square().sum() + eps)


def run():
    b, t, d = 7, 4, 5
    z = torch.randn(b, t, d, requires_grad=True)
    m = z.mean(1)
    r = z - m[:, None]
    vm = (m - m.mean(0)).square().mean()
    check("A_variance_decomposition", (z-z.mean((0, 1))).square().mean(),
          vm + r.square().mean())
    # Force the upper penalty active, so this is a nonzero-gradient check.
    upper = 0.01
    penalty = torch.relu(vm-upper).square()
    grad = torch.autograd.grad(penalty, z, retain_graph=True)[0]
    expected = (2*(vm-upper)*2*(m-m.mean(0))/(b*t*d))[:, None].expand_as(z)
    check("A_mean_penalty_gradient", grad, expected)
    check("A_mean_penalty_no_direct_residual_update", grad-grad.mean(1, keepdim=True),
          torch.zeros_like(grad))
    residual_loss = (r ** 4).mean()
    rg = torch.autograd.grad(residual_loss, z)[0]
    check("A_residual_gradient_time_sum", rg.sum(1), torch.zeros(b, d))
    static = torch.randn(b, 1, d).expand(b, t, d)
    check("A_static_window_exact_zero_residual", static-static.mean(1, keepdim=True),
          torch.zeros_like(static))

    m = torch.randn(b, d, requires_grad=True)
    c = covariance(m)
    eps = 1e-8
    q = participation(c, eps)
    dq = torch.autograd.grad(q, m)[0]
    tr, den = c.trace(), c.square().sum()+eps
    dc = 2*tr/den*torch.eye(d)-2*tr.square()/den.square()*c
    expected = 2/b*(m-m.mean(0)) @ dc
    check("Aplus_participation_gradient", dq, expected)
    assert torch.autograd.gradcheck(lambda a: participation(covariance(a)), (m,))
    RESULTS["Aplus_finite_difference"] = {"passed": True}
    rotation, _ = torch.linalg.qr(torch.randn(d, d))
    check("Aplus_rotation_invariance", q, participation(covariance(m @ rotation)))
    rank_one = torch.zeros(b, d)
    rank_one[:, 0] = torch.linspace(-2, 2, b)
    rank_one.requires_grad_()
    qr = participation(covariance(rank_one), eps=0)
    gr = torch.autograd.grad(torch.relu(1-qr/4).square(), rank_one)[0]
    check("Aplus_rank_one_positive_penalty", torch.relu(1-qr/4).square(), 0.5625)
    check("Aplus_rank_one_stationary_without_epsilon", gr, torch.zeros_like(gr))
    qeps = participation(covariance(rank_one), eps=eps)
    geps = torch.autograd.grad(torch.relu(1-qeps/4).square(), rank_one)[0]
    check("Aplus_rank_one_no_new_directions_with_epsilon", geps[:, 1:],
          torch.zeros_like(geps[:, 1:]))
    collapsed = torch.zeros(b, d, requires_grad=True)
    cc = covariance(collapsed)
    collapsed_loss = torch.relu(0.25-cc.trace()/d).square() + (1-participation(cc)/4).square()
    cg = torch.autograd.grad(collapsed_loss, collapsed)[0]
    check("Aplus_collapsed_stationary", cg, torch.zeros_like(cg))
    RESULTS["Aplus_collapsed_loss"] = float(collapsed_loss.detach())
    # Four nuisance coordinates alone can attain effective rank four.
    basis, _ = torch.linalg.qr(torch.randn(8, 4)-torch.randn(1, 4))
    centered = basis-basis.mean(0)
    basis, _ = torch.linalg.qr(centered)
    nuisance = torch.cat([basis*8**0.5, torch.zeros(8, 1)], dim=1)
    nc = covariance(nuisance)
    check("Aplus_nuisance_only_rank_four", participation(nc, eps=0), 4.0)

    tau = 0.2
    qv = torch.randn(d, requires_grad=True)
    keys = torch.randn(5, d, requires_grad=True)
    positive = torch.tensor([True, False, True, False, False])

    def b_loss(query, ys):
        scores = -(query-ys).square().sum(-1)/(d*tau)
        return scores.logsumexp(0)-scores[positive].logsumexp(0)

    scores = -(qv-keys).square().sum(-1)/(d*tau)
    posterior = scores.softmax(0)
    pos_posterior = scores[positive].softmax(0)
    qgrad = torch.autograd.grad(b_loss(qv, keys), qv)[0]
    qformula = 2/(d*tau)*((posterior[:, None]*keys).sum(0)
                          -(pos_posterior[:, None]*keys[positive]).sum(0))
    check("B_multi_positive_query_gradient", qgrad, qformula)
    assert torch.autograd.gradcheck(b_loss, (qv, keys))
    RESULTS["B_query_and_teacher_finite_difference"] = {"passed": True}
    constant_q = torch.zeros(d, requires_grad=True)
    constant_y = torch.zeros(5, d, requires_grad=True)
    constant_loss = b_loss(constant_q, constant_y)
    check("B_collapsed_loss_log_candidates_over_positives", constant_loss,
          torch.tensor(5/2).log())
    for i, value in enumerate(torch.autograd.grad(constant_loss, (constant_q, constant_y))):
        check(f"B_collapsed_gradient_{i}", value, torch.zeros_like(value))
    # An easy positive alone suffices; the other positive has negligible probability.
    pos_scores = torch.tensor([0.0, -1000.0, -50.0])
    weak_positive_loss = pos_scores.logsumexp(0)-pos_scores[:2].logsumexp(0)
    check("B_distant_second_positive_not_required", weak_positive_loss, 0.0)

    current = torch.randn(d, requires_grad=True)
    wp, wg = torch.randn(d, d), torch.randn(d, d)

    def gated(x):
        p = wp @ x
        g = (wg @ x).sigmoid()
        return x+g*(p-x)

    p, g = wp @ current, (wg @ current).sigmoid()
    jac = torch.autograd.functional.jacobian(gated, current)
    formula = torch.eye(d)+torch.diag(g)@(wp-torch.eye(d)) \
        + torch.diag(p-current)@torch.diag(g*(1-g))@wg
    check("C_total_jacobian", jac, formula)
    f = torch.randn(d)
    arbitrary_g = torch.rand(d)*0.8+0.1
    arbitrary_p = current+(f-current)/arbitrary_g
    check("C_unrestricted_predictor_reparameterization", current+arbitrary_g*(arbitrary_p-current), f)
    # Fixed proposal delta and hidden-feature group: least-squares optimal gate.
    proposal_delta = torch.tensor([1., 2., -1., 0.5])
    true_delta = torch.tensor([0.3, 1., 0., 0.1])
    optimum = (proposal_delta*true_delta).mean()/proposal_delta.square().mean()
    grid = torch.linspace(0, 1, 10001)
    grid_loss = (grid[:, None]*proposal_delta-true_delta).square().mean(1)
    check("C_conditional_optimal_gate", grid[grid_loss.argmin()], optimum.clamp(0, 1), atol=5e-5)
    # A small gate need not yield a stable state map: p=2z and constant g=.1.
    check("C_small_gate_expansive_counterexample", (1-0.1)+0.1*2, 1.1)
    RESULTS["C_small_gate_50_step_expansion"] = 1.1**50
    # Rotating diagonal gates generally cannot be represented by another diagonal gate.
    angle = torch.tensor(torch.pi/4)
    rot = torch.stack([torch.stack([angle.cos(), -angle.sin()]),
                       torch.stack([angle.sin(), angle.cos()])])
    rotated_gate = rot.T @ torch.diag(torch.tensor([0.1, 0.9])) @ rot
    check("C_diagonal_gate_not_rotation_equivariant", rotated_gate[0, 1].abs(), 0.4)
    print(json.dumps({"kind": "CPU algebra and counterexamples only", "torch": torch.__version__,
                      "checks": RESULTS}, indent=2))


if __name__ == "__main__":
    run()

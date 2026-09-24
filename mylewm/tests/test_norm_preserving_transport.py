"""Norm/inverse/derivative contracts, not claims of task success or convergence."""
import copy
import math

import pytest
import torch

from mylewm.algorithms.norm_preserving_transport import (
    NormPreservingTransport, NormPreservingSIGReg, dct2_basis, pair_mobius_change,
)
from mylewm.algorithms.objectives import GaussianSIGReg


def active_transport(dim=6, depth=2):
    torch.manual_seed(311)
    transport = NormPreservingTransport(dim=dim, depth=depth).double()
    with torch.no_grad():
        for block in transport.blocks:
            block.raw_vector.normal_(std=.7)
            block.orientation.angles.normal_(std=.1)
    return transport


@pytest.mark.parametrize('dim', [2, 6, 192])
def test_fixed_dct_basis_orthogonal_and_parameter_count(dim):
    basis = dct2_basis(dim)
    torch.testing.assert_close(basis @ basis.T, torch.eye(dim, dtype=torch.float64), rtol=1e-12, atol=5e-14)
    model = NormPreservingTransport(dim=dim)
    assert sum(p.numel() for p in model.parameters()) == 2 * (dim*(dim-1)//2 + dim)
    if dim == 192:
        assert sum(p.numel() for p in model.parameters()) == 37056


@pytest.mark.parametrize('depth', [0, 1, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_initial_map_is_bitwise_identity_including_origin(depth, dtype):
    model = NormPreservingTransport(dim=6, depth=depth).to(dtype=dtype)
    x = torch.randn(3, 4, 6, dtype=dtype)
    x[0, 0].zero_()
    torch.testing.assert_close(model(x), x, rtol=0, atol=0)
    torch.testing.assert_close(model.inverse(x), x, rtol=0, atol=0)


@pytest.mark.parametrize('dim,depth', [(2, 1), (6, 2), (192, 2)])
def test_nonzero_transport_preserves_all_radii_and_has_explicit_inverse(dim, depth):
    model = active_transport(dim, depth)
    values = torch.randn(17, dim, dtype=torch.float64)
    values = values * torch.logspace(-5, 3, 17, dtype=torch.float64)[:, None]
    values[0].zero_()
    result = model(values)
    torch.testing.assert_close(result.norm(dim=-1), values.norm(dim=-1), rtol=5e-12, atol=1e-12)
    torch.testing.assert_close(model.inverse(result), values, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(model(model.inverse(values)), values, rtol=1e-10, atol=1e-10)
    assert not torch.allclose(result[1:], values[1:])
    torch.testing.assert_close(model(torch.zeros(dim, dtype=torch.float64)), torch.zeros(dim, dtype=torch.float64),
                               rtol=0, atol=0)


def test_origin_jacobian_identity_parameter_gradcheck_and_empirical_bounds():
    model = active_transport(4)
    origin = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(model, origin)
    torch.testing.assert_close(jacobian, torch.eye(4, dtype=torch.float64), rtol=0, atol=0)
    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,), atol=2e-5, rtol=2e-4)
    names, parameters = zip(*model.named_parameters())
    def function(inputs, *weights):
        return torch.func.functional_call(model, dict(zip(names, weights)), (inputs,))
    assert torch.autograd.gradcheck(function, (x, *parameters), atol=2e-5, rtol=2e-4)
    singular = torch.linalg.svdvals(torch.autograd.functional.jacobian(model, x[0]))
    assert singular.min() >= model.lower_bound and singular.max() <= model.upper_bound
    before, after = torch.pdist(x), torch.pdist(model(x))
    assert bool((after >= model.lower_bound * before).all())
    assert bool((after <= model.upper_bound * before).all())


def test_not_volume_preserving_and_centered_variance_not_preserved():
    model = NormPreservingTransport(dim=2, depth=1).double()
    with torch.no_grad():
        model.blocks[0].raw_vector.copy_(torch.tensor([[1., 0.]], dtype=torch.float64))
    x = torch.tensor([.3, 1.], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(model, x)
    assert not torch.isclose(torch.linalg.det(jacobian), torch.tensor(1., dtype=torch.float64))
    batch = torch.tensor([[.5, 1.], [.5, -1.]], dtype=torch.float64)
    result = model(batch)
    torch.testing.assert_close(result.norm(dim=-1), batch.norm(dim=-1))
    assert not torch.isclose(batch.var(0, unbiased=False).sum(), result.var(0, unbiased=False).sum())
    # Norm preservation forbids positive isotropic gain on even one nonzero
    # point. It does not forbid arbitrary affine coincidences on finite data.
    assert not torch.allclose(result, 1.1 * batch)


def test_initial_sigreg_matches_raw_and_future_teacher_gets_gradient():
    torch.manual_seed(19)
    states = torch.randn(7, 4, 6, requires_grad=True)
    model = NormPreservingSIGReg(dim=6, projections=32)
    gaussian = GaussianSIGReg(dim=6, projections=32)
    torch.manual_seed(993)
    actual = model(states)
    torch.manual_seed(993)
    expected = gaussian(states.transpose(0, 1))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert states.grad[:, -1].norm() > 0
    for block in model.transport.blocks:
        assert block.raw_vector.grad.norm() > 0
        assert torch.count_nonzero(block.orientation.angles.grad) == 0
    assert not torch.allclose(model.transport.blocks[0].raw_vector.grad,
                              model.transport.blocks[1].raw_vector.grad)


@pytest.mark.parametrize('kind', ['constant', 'duplicates', 'n_less_than_d'])
def test_singular_sample_covariance_never_requires_inverse(kind):
    model = NormPreservingSIGReg(dim=192, projections=16)
    if kind == 'constant':
        states = torch.full((2, 4, 192), .25, requires_grad=True)
    elif kind == 'duplicates':
        states = torch.randn(1, 4, 192).expand(2, -1, -1).clone().requires_grad_()
    else:
        states = torch.randn(3, 4, 192, requires_grad=True)
    loss = model(states)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(states.grad).all()


def test_nonlinear_transport_can_correct_known_angular_distortion():
    # Constructed inverse image of Gaussian samples; existence witness only.
    torch.manual_seed(91)
    reference = torch.randn(2048, 4, 2)
    regularizer = NormPreservingSIGReg(dim=2, depth=1, projections=64)
    with torch.no_grad():
        regularizer.transport.blocks[0].raw_vector.copy_(torch.tensor([[1.2, -.8]]))
        source = regularizer.transport.inverse(reference)
    torch.manual_seed(117)
    transformed = regularizer(source)
    torch.manual_seed(117)
    untransformed = regularizer.gaussian(source.transpose(0, 1))
    assert transformed < untransformed
    torch.testing.assert_close(regularizer.transport(source).norm(dim=-1), source.norm(dim=-1),
                               rtol=1e-5, atol=1e-6)


def test_optimizer_checkpoint_exact_restore_and_orientation_can_learn(tmp_path):
    torch.manual_seed(7)
    model = NormPreservingSIGReg(dim=6, projections=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    states = torch.randn(11, 4, 6) + .3
    def update(current, opt):
        opt.zero_grad()
        loss = current(states)
        loss.backward()
        opt.step()
        return loss.detach()
    update(model, optimizer)
    path = tmp_path / 'regularizer.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'rng': torch.get_rng_state()}, path)
    expected = update(model, optimizer)
    assert any(b.orientation.angles.grad.norm() > 0 for b in model.transport.blocks)
    restored = NormPreservingSIGReg(dim=6, projections=32)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
    bundle = torch.load(path, weights_only=False)
    restored.load_state_dict(bundle['model'], strict=True)
    resumed_optimizer.load_state_dict(bundle['optimizer'])
    torch.set_rng_state(bundle['rng'])
    actual = update(restored, resumed_optimizer)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_zero_centrally_symmetric_gradient_is_documented_limitation():
    # Exact paired data at initialization can give zero vector gradient. This
    # score symmetry is not hidden by noise or a nonzero variance floor.
    torch.manual_seed(61)
    half = torch.randn(4, 4, 2)
    states = torch.cat((half, -half), dim=0)
    model = NormPreservingSIGReg(dim=2, projections=16)
    model(states).backward()
    assert all(block.raw_vector.grad.norm() < 2e-6 for block in model.transport.blocks)


@pytest.mark.parametrize('arguments', [dict(dim=3), dict(depth=3), dict(kappa=1), dict(kappa=-.1),
    dict(pair_size=4), dict(radial_unit=.01), dict(radial_unit=1), dict(basis_schedule='random')])
def test_invalid_or_unrecorded_recipe_choices_fail(arguments):
    with pytest.raises(ValueError):
        NormPreservingTransport(**arguments)


def test_autocast_still_uses_fp32_transport_and_statistics_are_explicit():
    regularizer = NormPreservingSIGReg(dim=6, projections=16)
    states = torch.randn(8, 4, 6)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = regularizer.transport(states.bfloat16())
        loss = regularizer(states)
    assert result.dtype == torch.float32 and loss.dtype == torch.float32
    stats = regularizer.statistics(states)
    assert stats['max_norm_absolute_error'] == 0
    assert stats['z_centered_variance_trace'] == stats['u_centered_variance_trace']
    config = regularizer.config()
    assert config['sample_pool'] == 'per_time_batch_not_pooled_four_frames'
    assert not config['transport']['centered_variance_preservation']

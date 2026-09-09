"""BT mathematics, native prediction, actual adapters, and short resume tests.

Synthetic inputs only: no benchmark training or success-rate claims.
"""
import argparse
import copy
import io
import json

import pytest
import torch

from mylewm.bt_sigreg import BoundedResidual, BoundedTransport, BTSIGReg, SpectralLinear
from mylewm.rbg import BlockSIGReg, one_step_objective
from mylewm import train_rbg as training
from mylewm.tests.test_training_state import assert_tree_equal


@pytest.mark.parametrize('kwargs', [dict(kappa=-.1), dict(kappa=1), dict(kappa=float('nan')),
                                   dict(depth=-1), dict(hidden=0), dict(dim=0)])
def test_reject_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        BoundedTransport(**kwargs)


def test_identity_initialization_can_start_learning():
    torch.manual_seed(17)
    t = BoundedTransport(dim=4, hidden=6, depth=2)
    z = torch.randn(3, 8, 4, requires_grad=True)
    torch.testing.assert_close(t(z), z, atol=0, rtol=0)
    t(z).square().mean().backward()
    assert all(b.output_weight.raw_spectrum.grad.abs().sum() > 0 for b in t.blocks)
    # Only the output weights can move on the first update; no dead all-zero MLP.
    opt = torch.optim.AdamW(t.parameters(), lr=.03)
    opt.step(); opt.zero_grad()
    assert not torch.equal(t(z), z)
    t(z).square().mean().backward()
    assert all(b.input_weight.raw_spectrum.grad.abs().sum() > 0 for b in t.blocks)
    assert all(b.bias.grad.abs().sum() > 0 for b in t.blocks)


@pytest.mark.parametrize('kappa', [0., .2, .8])
def test_bounds_origin_inverse_and_variance_after_updates(kappa):
    torch.manual_seed(11)
    t = BoundedTransport(dim=4, hidden=7, depth=3, kappa=kappa).double()
    opt = torch.optim.AdamW(t.parameters(), lr=2.)
    z = torch.randn(64, 4, dtype=torch.float64)
    for _ in range(4):
        opt.zero_grad(); (t(z)-2*z.roll(1, -1)).square().mean().backward(); opt.step()
        for b in t.blocks:
            for w in (b.input_weight, b.output_weight):
                assert torch.linalg.matrix_norm(w(), 2) <= 1 + 1e-12
        u = t(z)
        dz, du = torch.pdist(z), torch.pdist(u)
        assert torch.all(du >= t.lower_bound * dz - 1e-10)
        assert torch.all(du <= t.upper_bound * dz + 1e-10)
        jac = torch.autograd.functional.jacobian(t, z[0])
        singular = torch.linalg.svdvals(jac)
        assert singular.min() >= t.lower_bound - 1e-10
        assert singular.max() <= t.upper_bound + 1e-10
        vz, vu = z.var(0, unbiased=False).sum(), u.var(0, unbiased=False).sum()
        assert vu/t.upper_bound**2 <= vz + 1e-10
        assert vz <= vu/t.lower_bound**2 + 1e-10
    torch.testing.assert_close(t(torch.zeros_like(z)), torch.zeros_like(z), atol=0, rtol=0)
    # Banach iteration in tests only; neither inverse nor logdet is in the loss.
    recovered = u.detach()
    with torch.no_grad():
        for b in reversed(t.blocks):
            target, x = recovered, recovered.clone()
            for _ in range(160):
                x = target - (b(x)-x)
            recovered = x
    torch.testing.assert_close(recovered, z, atol=1e-9, rtol=1e-9)


def test_no_time_batch_or_train_eval_conditioning_and_gradcheck():
    torch.manual_seed(3)
    t = BoundedTransport(dim=3, hidden=4).double()
    with torch.no_grad():
        for b in t.blocks: b.output_weight.raw_spectrum.normal_()
    z = torch.randn(2, 5, 3, dtype=torch.float64)
    expected = t(z)
    torch.testing.assert_close(t(z.flip(1)).flip(1), expected)
    torch.testing.assert_close(t(z.flatten(0, 1)).reshape_as(z), expected)
    torch.testing.assert_close(t(z[:, :1]), expected[:, :1])
    t.eval(); torch.testing.assert_close(t(z), expected, atol=0, rtol=0)
    assert torch.autograd.gradcheck(t, (z.requires_grad_(),))
    b = t.blocks[0]
    names, parameters = zip(*b.named_parameters())
    def function(x, *params):
        return torch.func.functional_call(b, dict(zip(names, params)), (x,))
    assert torch.autograd.gradcheck(function, (z, *parameters))


@pytest.mark.parametrize('input_dim,output_dim', [(4, 4), (3, 7), (7, 3)])
def test_spectral_matrices_are_bounded_and_rectangular(input_dim, output_dim):
    w = SpectralLinear(input_dim, output_dim, .8).double()
    with torch.no_grad():
        for p in w.parameters(): p.normal_(std=2.)
    matrix = w()
    assert matrix.shape == (output_dim, input_dim)
    for q in (w.left(), w.right()):
        torch.testing.assert_close(q.T @ q, torch.eye(len(q), dtype=q.dtype), atol=1e-12, rtol=1e-12)
    assert torch.linalg.matrix_norm(matrix, 2) <= 1+1e-12


def test_transport_can_break_oddness_and_deform_all_directions():
    # Explicit witness, not a claim that training discovers this configuration.
    dim = 12
    t = BoundedTransport(dim=dim, hidden=dim, depth=2, kappa=.2).double()
    with torch.no_grad():
        for block in t.blocks:
            for w in (block.input_weight, block.output_weight):
                w.left.angles.zero_(); w.right.angles.zero_()
                w.raw_spectrum.fill_(torch.atanh(torch.tensor(.8)).item())
            block.bias.fill_(.5)
    z = torch.ones(dim, dtype=torch.float64)
    assert (t(z)+t(-z)).abs().min() > 1e-3
    torch.testing.assert_close(t(torch.zeros_like(z)), torch.zeros_like(z), atol=0, rtol=0)
    jac = torch.autograd.functional.jacobian(t, torch.zeros_like(z))
    singular = torch.linalg.svdvals(jac)
    assert torch.all(singular > 1.1)
    # The previous Frobenius prototype forbade nuclear difference > .44.
    assert torch.linalg.matrix_norm(jac-torch.eye(dim), 'nuc') > .44
    assert singular.max() < t.upper_bound
    # Linear activation component permits asymptotically nonzero relative displacement.
    far = 1e5*z
    assert (t(far)-far).norm()/far.norm() > .1


def test_new_parameterization_metadata_and_parameter_count():
    t = BoundedTransport()
    assert sum(p.numel() for p in t.parameters()) == 147840
    assert t.config()['normalization'] == 'cayley_spectral_v2'


class TinyBTModel(torch.nn.Module):
    def __init__(self, views=1):
        super().__init__()
        self.views = views
        self.encoder = torch.nn.Sequential(torch.nn.Linear(3*views, 192), torch.nn.Dropout(.2))
        self.projector = torch.nn.Linear(192, 192)
        self.action_encoder = torch.nn.Linear(10 if views == 1 else 28, 192)
        self.predictor = torch.nn.Linear(192, 192)

    def encode(self, info):
        x = info['pixels'].flatten(2)
        return {'emb': self.projector(self.encoder(x))}

    def predict(self, z, a):
        return self.predictor(z+a)


@pytest.mark.parametrize('depth', [0, 2])
def test_identity_matches_raw_loss_and_all_world_gradients(depth):
    torch.manual_seed(9)
    model = TinyBTModel().eval()
    x, a = torch.randn(3, 4, 3, requires_grad=True), torch.randn(3, 3, 10)
    raw, bt = BlockSIGReg(blocks=1), BTSIGReg(depth=depth)
    torch.manual_seed(21)
    loss, parts = one_step_objective(model, x, a, raw, cross_weight=0, mode='raw')
    grads = torch.autograd.grad(loss, [x, *model.parameters()])
    torch.manual_seed(21)
    other, btparts = one_step_objective(model, x, a, bt, cross_weight=0, mode='bt')
    btgrads = torch.autograd.grad(other, [x, *model.parameters()])
    torch.testing.assert_close(loss, other, atol=0, rtol=0)
    for first, second in zip(grads, btgrads):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert parts.keys() == btparts.keys()


def test_prediction_is_native_and_future_teacher_has_gradient():
    model = TinyBTModel().eval()
    x, a = torch.randn(3, 4, 3, requires_grad=True), torch.randn(3, 3, 10)
    reg = BTSIGReg()
    with torch.no_grad():
        for block in reg.transport.blocks: block.output_weight.raw_spectrum.normal_()
    seen = []
    hook = reg.transport.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]))
    loss, parts = one_step_objective(model, x, a, reg, cross_weight=0, mode='bt')
    hook.remove()
    z = model.encode({'pixels': x})['emb']
    expected = (model.predict(z[:, :3], model.action_encoder(a))-z[:, 1:]).square().mean()
    torch.testing.assert_close(parts['prediction'], expected, atol=0, rtol=0)
    torch.testing.assert_close(seen[0], z.transpose(0, 1))
    teacher_grad = torch.autograd.grad(parts['prediction'], x, retain_graph=True)[0]
    assert teacher_grad[:, -1].abs().sum() > 0
    assert all(g is None for g in torch.autograd.grad(parts['prediction'], reg.parameters(),
                                                     allow_unused=True, retain_graph=True))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in reg.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.encoder.parameters())


class BTClips(torch.utils.data.Dataset):
    def __init__(self, manifest, validation=False):
        self.views, self.n = manifest['views'], 4 if validation else 12
    def __len__(self): return self.n
    def __getitem__(self, index):
        x = torch.arange(4*self.views*3).float().reshape(4, self.views, 3)/20+index/12
        if self.views == 1: x = x[:, 0]
        return x, torch.full((3, 10 if self.views == 1 else 28), index/12)


@pytest.mark.parametrize('views', [1, 2])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bt_actual_training_loop_resume_and_inference_export(tmp_path, monkeypatch, views, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    if device == 'cpu': monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(training, 'build_model', lambda: TinyBTModel(views))
    monkeypatch.setattr(training, 'Clips', BTClips)
    monkeypatch.setattr(training, 'preprocess', lambda batch, m, d: tuple(x.to(d) for x in batch))
    dataset = tmp_path/'synthetic.identity'; dataset.write_bytes(b'fixed synthetic clips')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(dataset=str(dataset), dataset_size=dataset.stat().st_size,
        dataset_mtime_ns=dataset.stat().st_mtime_ns, action_mean=[0.], action_std=[1.], views=views)))
    args = argparse.Namespace(steps=4, warmup_steps=1, lr=.003, min_lr=0., seed=17,
        manifest=manifest, mode='bt', blocks=4, batch_size=4, workers=0, bt_depth=2,
        bt_hidden=12, bt_kappa=.3, gaussian_weight=.09, cross_weight=.01,
        diagnostics_every=1, save_every=2, resume=False, output=tmp_path/'full', deterministic=True)
    training.train(args)
    full = torch.load(args.output/'resume.pt', map_location='cpu', weights_only=False)
    initial = torch.load(args.output/'initialization.pt', map_location='cpu', weights_only=False)
    keys = [k for k in full['regularizer'] if 'output_weight.raw_spectrum' in k]
    assert keys and all(not torch.equal(full['regularizer'][k], initial['regularizer'][k]) for k in keys)
    assert len(full['optimizer']['param_groups']) == 2
    saved = torch.load(args.output/'step_4_object.ckpt', map_location='cpu', weights_only=False)
    assert not any(isinstance(m, BoundedTransport) for m in saved.modules())
    assert_tree_equal(saved.state_dict(), full['model'])
    assert all('transport' not in key for key in full['model'])
    original = training.one_step_objective
    calls = 0
    def interrupt(*pos, **kw):
        nonlocal calls
        if pos[0].training:
            calls += 1
            if calls == 3: raise InterruptedError('checkpoint saved at step 2')
        return original(*pos, **kw)
    args.output = tmp_path/'resumed'
    monkeypatch.setattr(training, 'one_step_objective', interrupt)
    with pytest.raises(InterruptedError): training.train(args)
    monkeypatch.setattr(training, 'one_step_objective', original)
    args.resume = True; training.train(args)
    resumed = torch.load(args.output/'resume.pt', map_location='cpu', weights_only=False)
    for key in ('model', 'regularizer', 'optimizer', 'scheduler', 'random_state', 'next_batch_index'):
        assert_tree_equal(full[key], resumed[key])
    args.bt_kappa = .4
    with pytest.raises(ValueError, match='resume mismatch'): training.train(args)


@pytest.mark.parametrize('benchmark', ['pusht', 'libero10'])
def test_real_architecture_update_and_native_planning_export(benchmark):
    from types import SimpleNamespace
    from train import lejepa_forward
    from module import SIGReg
    from mylewm.train_rbg_libero import build_model as build_libero
    torch.set_num_threads(4); torch.manual_seed(55)
    model = (training.build_model() if benchmark == 'pusht' else build_libero()).train()
    raw = copy.deepcopy(model)
    reg = BTSIGReg(hidden=16)
    shape = (2, 4, 3, 28, 28) if benchmark == 'pusht' else (2, 4, 2, 3, 28, 28)
    x = torch.randn(shape)
    a = torch.randn(2, 3, 10 if benchmark == 'pusht' else 28)
    opt = torch.optim.AdamW([{'params':model.parameters()}, {'params':reg.parameters()}], lr=1e-3)
    # Direct pinned upstream comparison: finite actions, CPU FP32, same RNG/lambda.
    official = SimpleNamespace(model=raw, sigreg=SIGReg(), log_dict=lambda *args, **kw: None)
    cfg = SimpleNamespace(history_size=3, num_preds=1,
                          loss=SimpleNamespace(sigreg=SimpleNamespace(weight=.09)))
    torch.manual_seed(991)
    initial_loss, _ = one_step_objective(model, x, a, reg, cross_weight=0, mode='bt')
    initial_loss.backward()
    torch.manual_seed(991)
    raw_loss = lejepa_forward(official, {'pixels': x, 'action': a}, 'train', cfg)['loss']
    raw_loss.backward()
    torch.testing.assert_close(initial_loss, raw_loss, atol=0, rtol=0)
    for p, q in zip(model.parameters(), raw.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
    before = copy.deepcopy(reg.state_dict())
    for _ in range(2):
        opt.zero_grad()
        loss, parts = one_step_objective(model, x, a, reg, cross_weight=0, mode='bt')
        assert torch.isfinite(loss)
        loss.backward(); opt.step()
    assert any(not torch.equal(before[k], v) for k, v in reg.state_dict().items()
               if 'output_weight.raw_spectrum' in k)
    assert model.state_dict().keys() == raw.state_dict().keys()
    model.eval()
    buffer = io.BytesIO(); torch.save(model, buffer); buffer.seek(0)
    exported = torch.load(buffer, weights_only=False)
    info = dict(pixels=x[:1, :3].unsqueeze(1), goal=x[:1, 1:].unsqueeze(1),
                action=a[:1, :1].unsqueeze(1))
    candidates = torch.randn(1, 2, 4, a.shape[-1])
    def forbidden(*args): raise AssertionError('Planning must never call T')
    reg.transport.forward = forbidden
    with torch.no_grad():
        expected = model.get_cost(copy.deepcopy(info), candidates)
        actual = exported.get_cost(copy.deepcopy(info), candidates)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    assert not any(isinstance(m, BoundedTransport) for m in exported.modules())


def test_both_legacy_cli_adapters():
    import os
    import subprocess
    import sys
    for name in ('train_rbg.py', 'train_rbg_libero.py'):
        result = subprocess.run([sys.executable, str(training.ROOT/'mylewm'/name), '--help'],
                                capture_output=True, text=True, timeout=30,
                                env={k:v for k,v in os.environ.items() if k != 'PYTHONPATH'})
        assert result.returncode == 0, result.stderr
        assert '--bt-kappa' in result.stdout and 'bt}' in result.stdout


def test_bt_precision_and_cross_loss_rejection():
    model = TinyBTModel().eval()
    reg = BTSIGReg()
    x, a = torch.randn(2, 4, 3), torch.randn(2, 3, 10)
    with pytest.raises(ValueError, match='cross_weight'):
        one_step_objective(model, x, a, reg, mode='bt')
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        u = reg.transport(torch.randn(4, 2, 192).bfloat16())
        loss, _ = one_step_objective(model, x, a, reg, cross_weight=0, mode='bt')
    assert u.dtype == torch.float32 and loss.dtype == torch.float32
    assert torch.isfinite(loss)

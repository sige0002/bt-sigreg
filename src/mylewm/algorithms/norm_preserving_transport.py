"""Training-only pairwise Möbius transport with exact real-arithmetic radii.

Each smooth two-coordinate map preserves its pair radius. Orthogonal
conjugations mix the pairs; the full shared transport preserves ||z|| for
EVERY input. It therefore cannot supply a nonunit positive isotropic gain,
even just on a finite collection of nonzero states. This is not preservation
of centered variance: angular motion can change the mean and centered spread.
Nor does it exclude every affine fit on a finite dataset, or guarantee useful
nonlinearity, task information, optimization or control performance.

The unit in sqrt(1+||x_pair||²) is an explicit Gaussian-reference coordinate
choice, not a hidden epsilon. The parameter bound is the declared κ. No batch
statistics, state noise, sample-covariance inverse or data-dependent clipping
enters T. Radius preservation/invertibility/bounds hold in real arithmetic;
FP32/FP64 checks are numerical evidence, not interval certificates.

Sphere Möbius maps follow Rezende et al. (2020), 'Normalizing Flows on Tori
and Spheres'. The smooth radius extension, independent coordinate pairs and
orthogonal mixing are this recipe's design choices. Unlike composition on
one fixed full-dimensional sphere, the mixed pair construction does not
remain a single global sphere Möbius family.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mylewm.algorithms.bt_sigreg import CayleyOrthogonal
from mylewm.algorithms.objectives import GaussianSIGReg


def _validate(dim, depth, kappa, pair_size, basis_schedule, radial_unit):
    if type(dim) is not int or dim < 2 or dim % 2:
        raise ValueError('Require an even integer dimension >= 2')
    if type(depth) is not int or depth not in (0, 1, 2):
        raise ValueError('This explicit basis schedule supports depth 0, 1 or 2')
    if (isinstance(kappa, bool) or not isinstance(kappa, (int, float))
            or not math.isfinite(kappa) or not 0 <= kappa < 1):
        raise ValueError('Require finite 0 <= kappa < 1')
    if type(pair_size) is not int or pair_size != 2:
        raise ValueError('pair_size must be 2, the smallest nontrivial smooth sphere map')
    if basis_schedule != 'identity_then_dct2':
        raise ValueError('Require explicit identity_then_dct2 basis schedule')
    if type(radial_unit) is not float or radial_unit != 1.0:
        raise ValueError('radial_unit must be the explicit Gaussian-reference value 1.0')


def dct2_basis(dim):
    """Orthonormal DCT-II matrix in CPU FP64, defined solely by dimension."""
    if type(dim) is not int or dim < 1:
        raise ValueError('DCT dimension must be a positive integer')
    k = torch.arange(dim, dtype=torch.float64)[:, None]
    j = torch.arange(dim, dtype=torch.float64)[None, :]
    result = math.sqrt(2 / dim) * torch.cos(math.pi / dim * (j + .5) * k)
    result[0] /= math.sqrt(2)
    return result


def pair_mobius_change(pairs, vector):
    """Return F_v(x)-x for [...,pair,2] and bounded v[pair,2].

The residual form is algebraically equivalent to the Möbius fraction and
returns bitwise zero at v=0. Origin smoothness needs no division by ||x||.
"""
    if (not isinstance(pairs, torch.Tensor) or pairs.ndim < 2 or pairs.shape[-1] != 2
            or not all(pairs.shape) or not isinstance(vector, torch.Tensor)
            or vector.shape != pairs.shape[-2:] or vector.device != pairs.device
            or vector.dtype != pairs.dtype or pairs.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Require matching float32/float64 pairs [...,P,2] and vector[P,2]')
    if not torch.isfinite(pairs).all() or not torch.isfinite(vector).all():
        raise ValueError('Möbius arguments must be finite')
    if not bool((vector.square().sum(-1) < 1).all()):
        raise ValueError('Möbius vector must lie strictly inside the unit disk')
    squared_radius = pairs.square().sum(-1, keepdim=True)
    inverse_scale = torch.rsqrt(1 + squared_radius)
    q = vector * inverse_scale
    p = (q * pairs).sum(-1, keepdim=True)
    a = squared_radius * q.square().sum(-1, keepdim=True)
    denominator = 1 + 2*p + a
    change = 2 * (squared_radius * (1+p) * q - (a+p) * pairs) / denominator
    if (not torch.isfinite(change).all() or not torch.isfinite(denominator).all()
            or not bool((denominator > 0).all())):
        raise FloatingPointError('Nonfinite Möbius arithmetic or nonpositive denominator')
    return change


class _PairMobiusBlock(nn.Module):
    def __init__(self, dim, kappa, fixed_basis):
        super().__init__()
        self.dim, self.kappa = dim, float(kappa)
        self.orientation = CayleyOrthogonal(dim)
        # Explicit identity initialization of the learned orientation. The
        # fixed bases differ between layers, so their vector gradients need
        # not coincide despite the initial complete map being identity.
        nn.init.zeros_(self.orientation.angles)
        self.raw_vector = nn.Parameter(torch.zeros(dim // 2, 2))
        self.register_buffer('fixed_basis', fixed_basis.clone())

    def bounded_vector(self, dtype):
        raw = self.raw_vector.to(dtype)
        squared_norm = raw.square().sum(-1, keepdim=True)
        if not torch.isfinite(squared_norm).all():
            raise FloatingPointError('Nonfinite raw Möbius parameter norm')
        value = self.kappa * raw / torch.sqrt(1 + squared_norm)
        if not torch.isfinite(value).all():
            raise FloatingPointError('Nonfinite bounded Möbius vector')
        return value

    def forward(self, states, orthogonal, *, inverse=False):
        matrix = orthogonal.to(states.dtype) @ self.fixed_basis.to(states.dtype)
        rotated = F.linear(states, matrix)
        pairs = rotated.reshape(*rotated.shape[:-1], self.dim // 2, 2)
        vector = self.bounded_vector(states.dtype)
        if inverse:
            vector = -vector
        delta = pair_mobius_change(pairs, vector).reshape_as(rotated)
        # Real-arithmetic equivalent of Q^T H(Qx), avoiding Q^TQ rounding
        # drift at zero vectors. Nonzero-vector numerical errors remain.
        return states + F.linear(delta, matrix.T)


class NormPreservingTransport(nn.Module):
    def __init__(self, dim=192, depth=2, kappa=.2, pair_size=2,
                 basis_schedule='identity_then_dct2', radial_unit=1.0):
        super().__init__()
        _validate(dim, depth, kappa, pair_size, basis_schedule, radial_unit)
        self.dim, self.depth, self.kappa = dim, depth, float(kappa)
        self.pair_size, self.basis_schedule, self.radial_unit = pair_size, basis_schedule, radial_unit
        bases = (torch.eye(dim, dtype=torch.float64), dct2_basis(dim))
        self.blocks = nn.ModuleList(_PairMobiusBlock(dim, self.kappa, bases[i]) for i in range(depth))
        self.register_buffer('_compute_precision', torch.empty(0, dtype=torch.float32), persistent=False)
        # A conservative analytic bound for the smooth pair map and its
        # inverse; orthogonal conjugation preserves it. This is not a fitted
        # singular-value estimate. T's radius preservation is stronger than
        # this bound specifically for distances from zero.
        self.block_lipschitz_bound = (1 + 6*self.kappa + 4*self.kappa**2) / (1-self.kappa)**2
        self.upper_bound = self.block_lipschitz_bound**depth
        self.lower_bound = 1 / self.upper_bound
        if not math.isfinite(self.upper_bound) or self.lower_bound <= 0:
            raise ValueError('Transport bound overflow/underflow')

    def _transform(self, states, inverse):
        dtype = self._compute_precision.dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Transport parameters must retain float32 or float64 precision')
        if any(parameter.dtype != dtype for parameter in self.parameters()):
            raise ValueError('Transport parameters have inconsistent precision')
        if (not isinstance(states, torch.Tensor) or states.ndim < 1 or not all(states.shape)
                or states.shape[-1] != self.dim or not states.is_floating_point()
                or states.device != self._compute_precision.device or not torch.isfinite(states).all()):
            raise ValueError('Require finite floating states [...,dim] on the transport device')
        with torch.autocast(device_type=states.device.type, enabled=False):
            values = states.to(dtype)
            factors = CayleyOrthogonal.batched(block.orientation for block in self.blocks)
            blocks = reversed(self.blocks) if inverse else self.blocks
            for block in blocks:
                values = block(values, factors[block.orientation], inverse=inverse)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite norm-preserving transport output')
        return values

    def forward(self, states):
        return self._transform(states, False)

    def inverse(self, states):
        return self._transform(states, True)

    def config(self):
        return {'family': 'mixed_pair_mobius_norm_preserving_v1',
                'dim': self.dim, 'depth': self.depth, 'kappa': self.kappa,
                'pair_size': self.pair_size, 'basis_schedule': self.basis_schedule,
                'radial_unit': self.radial_unit, 'radial_unit_meaning': 'Gaussian-reference coordinate unit',
                'parameterization': 'kappa_w_div_sqrt_1_plus_squared_norm',
                'orientation': 'learned_Cayley_times_fixed_basis',
                'initialization': 'zero_vectors_zero_Cayley_angles_identity_transport',
                'fixed_basis_storage_dtype': str(self.blocks[0].fixed_basis.dtype) if self.blocks else None,
                'compute_dtype': str(self._compute_precision.dtype),
                'norm_preservation': 'pointwise_real_arithmetic',
                'centered_variance_preservation': False,
                'sample_covariance_inverse': False,
                'block_lipschitz_bound': self.block_lipschitz_bound,
                'lower_bound': self.lower_bound, 'upper_bound': self.upper_bound,
                'bound_scope': 'real_arithmetic_not_floating_point_certificate'}


class NormPreservingSIGReg(nn.Module):
    def __init__(self, dim=192, depth=2, kappa=.2, projections=1024, knots=17,
                 pair_size=2, basis_schedule='identity_then_dct2', radial_unit=1.0):
        super().__init__()
        self.dim = dim
        self.transport = NormPreservingTransport(dim=dim, depth=depth, kappa=kappa,
            pair_size=pair_size, basis_schedule=basis_schedule, radial_unit=radial_unit)
        self.gaussian = GaussianSIGReg(dim=dim, projections=projections, knots=knots)

    def forward(self, states):
        if (not isinstance(states, torch.Tensor) or states.ndim != 3
                or states.shape[0] < 2 or states.shape[1:] != (4, self.dim)):
            raise ValueError('Norm-preserving SIGReg requires states [B>=2,4,dim]')
        return self.gaussian(self.transport(states).transpose(0, 1))

    @torch.no_grad()
    def statistics(self, states):
        if not isinstance(states, torch.Tensor) or states.ndim != 3 or states.shape[1:] != (4, self.dim):
            raise ValueError('Diagnostics require states [B,4,dim]')
        transported = self.transport(states)
        z, u = states.detach().double(), transported.double()
        original_norm, result_norm = z.norm(dim=-1), u.norm(dim=-1)
        return {'max_norm_absolute_error': float((original_norm-result_norm).abs().max()),
                'z_mean_square': float(z.mean(0).square().mean()),
                'u_mean_square': float(u.mean(0).square().mean()),
                'z_centered_variance_trace': float(z.var(0, unbiased=False).sum(-1).mean()),
                'u_centered_variance_trace': float(u.var(0, unbiased=False).sum(-1).mean()),
                'displacement_rms': float((u-z).square().mean().sqrt())}

    def config(self):
        return {'algorithm': 'norm_preserving_gaussian_sigreg_v1', 'dim': self.dim,
                'transport': self.transport.config(), 'regularizer': 'original_GaussianSIGReg',
                'sample_pool': 'per_time_batch_not_pooled_four_frames',
                'projections': self.gaussian.projections, 'knots': int(self.gaussian.t.numel()),
                'score_dtype': 'torch.float32', 'student_scale_invariance': False}

"""Training-only bounded transport; prediction and planning remain in native z.

For g(x)=kappa B [phi(Ax+b)-phi(b)], phi(x)=(x+tanh(x))/2 is
1-Lipschitz. A and B have orthogonal Cayley factors and bounded singular
values, giving operator norm <= 1 without a Frobenius budget. Thus
Lip(g) <= kappa and I+g is globally invertible. The proof is
in real arithmetic; the safety factor and numerical tests do not constitute
an interval-arithmetic certificate for floating-point execution.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mylewm.objectives import GaussianSIGReg


def add_bt_arguments(parser):
    parser.add_argument('--bt-depth', type=int, default=2,
                        help='Residual transport blocks; 0 is the identity control')
    parser.add_argument('--bt-kappa', type=float, default=.2,
                        help='Per-block Lipschitz budget, 0 <= kappa < 1; not a tuned value')
    parser.add_argument('--bt-hidden', type=int, default=192)


class CayleyOrthogonal(nn.Module):
    """Packed skew parameters: a finite Cayley chart, not all rotations globally."""
    def __init__(self, dim):
        super().__init__()
        indices = torch.triu_indices(dim, dim, offset=1)
        self.register_buffer('indices', indices, persistent=False)
        self.angles = nn.Parameter(torch.randn(indices.shape[1]) * (.01 / math.sqrt(dim)))
        self.dim = dim

    def forward(self):
        upper = self.angles.new_zeros(self.dim, self.dim)
        upper = upper.index_put(tuple(self.indices), self.angles)
        skew = upper - upper.T
        identity = torch.eye(self.dim, device=skew.device, dtype=skew.dtype)
        return torch.linalg.solve(identity-skew, identity+skew)


class SpectralLinear(nn.Module):
    """Rectangular Q_left diag(tanh(s)) Q_right.T, ||W||_2 <= 1."""
    def __init__(self, input_dim, output_dim, initial_singular_value):
        super().__init__()
        self.left = CayleyOrthogonal(output_dim)
        self.right = CayleyOrthogonal(input_dim)
        self.rank = min(input_dim, output_dim)
        self.raw_spectrum = nn.Parameter(torch.full(
            (self.rank,), math.atanh(initial_singular_value)))

    def forward(self):
        left = self.left()[:, :self.rank]
        right = self.right()[:, :self.rank]
        return (left * self.raw_spectrum.tanh()) @ right.T


class BoundedResidual(nn.Module):
    def __init__(self, dim, hidden, kappa):
        super().__init__()
        if dim < 1 or hidden < 1 or not math.isfinite(kappa) or not 0 <= kappa < 1:
            raise ValueError('Positive dimensions and finite 0 <= kappa < 1 required')
        self.kappa = float(kappa)
        self.input_weight = SpectralLinear(dim, hidden, .8)
        self.output_weight = SpectralLinear(hidden, dim, 0.)
        self.bias = nn.Parameter(torch.empty(hidden).uniform_(-.5, .5))

    @staticmethod
    def activation(x):
        return .5*x + .5*torch.tanh(x)

    def forward(self, x):
        a, b = self.input_weight(), self.output_weight()
        # Evaluate the linear difference analytically to avoid cancellation of bias.
        ax = F.linear(x, a)
        centered = .5*ax + .5*(torch.tanh(ax+self.bias)-torch.tanh(self.bias))
        return x + self.kappa * (1 - 1e-4) * F.linear(centered, b)


class BoundedTransport(nn.Module):
    def __init__(self, dim=192, hidden=192, depth=2, kappa=.2):
        super().__init__()
        if not isinstance(depth, int) or depth < 0:
            raise ValueError('depth must be a nonnegative integer')
        if dim < 1 or hidden < 1 or not math.isfinite(kappa) or not 0 <= kappa < 1:
            raise ValueError('Positive dimensions and finite 0 <= kappa < 1 required')
        self.dim, self.hidden, self.depth, self.kappa = dim, hidden, depth, float(kappa)
        self.lower_bound, self.upper_bound = (1-kappa)**depth, (1+kappa)**depth
        if not 0 < self.lower_bound <= self.upper_bound < float('inf'):
            raise ValueError('Transport bounds underflow/overflow')
        self.blocks = nn.ModuleList([BoundedResidual(dim, hidden, kappa) for _ in range(depth)])

    def forward(self, z):
        if z.shape[-1] != self.dim:
            raise ValueError('Transport dimension mismatch')
        # FP64 is retained for mathematical tests; mixed-precision training uses FP32.
        with torch.autocast(device_type=z.device.type, enabled=False):
            u = z if z.dtype == torch.float64 else z.float()
            for block in self.blocks:
                u = block(u)
        return u

    def config(self):
        return dict(dim=self.dim, hidden=self.hidden, depth=self.depth, kappa=self.kappa,
                    lower_bound=self.lower_bound, upper_bound=self.upper_bound,
                    normalization='cayley_spectral_v2', safety_factor=1-1e-4,
                    activation='centered_half_linear_half_tanh_learned_bias',
                    initialization='identity_zero_output_spectrum_input_spectrum_0.8')


class BTSIGReg(nn.Module):
    def __init__(self, dim=192, hidden=192, depth=2, kappa=.2, projections=1024, knots=17):
        super().__init__()
        self.transport = BoundedTransport(dim, hidden, depth, kappa)
        self.gaussian = GaussianSIGReg(dim=dim, projections=projections, knots=knots)

    def forward(self, z):
        return self.gaussian(self.transport(z))

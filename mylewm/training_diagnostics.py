"""Sparse diagnostics; these quantities do not contribute to the training loss."""
import torch


@torch.no_grad()
def transport_statistics(z, u):
    """Detached, per-time batch statistics; no new forward or random draws.

    Finite-batch spectra and sampled distances are diagnostics, not certificates.
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        z, u = z.detach().float(), u.detach().float()
        if z.shape != u.shape or z.ndim != 3 or z.shape[1] < 2:
            raise ValueError('Expected matching (time, batch >= 2, dim) tensors')
        result = {}
        for name, value in [('z', z), ('u', u)]:
            centered = value - value.mean(1, keepdim=True)
            covariance = centered.transpose(1, 2) @ centered / (value.shape[1] - 1)
            eigenvalues = torch.linalg.eigvalsh(covariance)
            result.update({f'{name}_variance_trace': float(covariance.diagonal(dim1=-2, dim2=-1).sum(-1).mean()),
                           f'{name}_cov_eigen_min': float(eigenvalues.min()),
                           f'{name}_cov_eigen_max': float(eigenvalues.max()),
                           f'{name}_mean_square': float(value.mean(1).square().mean())})
        result['displacement_rms'] = float((u-z).square().mean().sqrt())
        result['relative_displacement_rms'] = float((u-z).norm() / z.norm().clamp_min(1e-12))
        distances = (z-z.roll(1, 1)).norm(dim=-1)
        valid = distances > 1e-12
        ratios = (u-u.roll(1, 1)).norm(dim=-1)[valid] / distances[valid]
        result['sampled_distance_pairs'] = int(valid.sum())
        result['sampled_distance_ratio_min'] = float(ratios.min()) if ratios.numel() else None
        result['sampled_distance_ratio_max'] = float(ratios.max()) if ratios.numel() else None
        return result


def encoder_gradient_norms(model,parts,gaussian_weight,cross_weight):
    params=[p for component in (model.encoder,model.projector) for p in component.parameters() if p.requires_grad]
    result={}
    for name,weight in [('prediction',1.),('gaussian',gaussian_weight),('cross',cross_weight)]:
        term=parts[name]*weight
        if not term.requires_grad:
            result[name]=0.;continue
        gradients=torch.autograd.grad(term,params,retain_graph=True,allow_unused=True)
        total=sum(float(g.detach().float().square().sum()) for g in gradients if g is not None)
        result[name]=total**.5
    return result


@torch.no_grad()
def action_diagnostics(model,pixels,actions):
    if model.training:raise ValueError('Action diagnostics require eval mode')
    z=model.encode({'pixels':pixels})['emb']
    reference=model.predict(z[:,:3],model.action_encoder(actions))
    variants={'shuffled_batch':actions.roll(1,0),'zero_normalized_action':torch.zeros_like(actions)}
    if hasattr(model,'training_action_mean'):
        d=model.training_action_mean.numel()
        zero=(-model.training_action_mean/model.training_action_std).to(actions)
        variants['zero_physical_action']=zero.repeat(actions.shape[-1]//d).expand_as(actions)
    return {name:{'prediction_mse':float((pred-z[:,1:]).square().mean()),
                  'change_from_reference_mse':float((pred-reference).square().mean())}
            for name,a in variants.items()
            for pred in [model.predict(z[:,:3],model.action_encoder(a))]}

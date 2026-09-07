"""Sparse diagnostics; these quantities do not contribute to the training loss."""
import torch


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

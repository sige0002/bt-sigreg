"""BN-only diagnostic routes and eval-mode CEM candidate invariance."""
import torch


def bn_statistics(model):
    result={}
    handles=[]
    for name,layer in model.named_modules():
        if not isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):continue
        def capture(module,inputs,key=name):
            x=inputs[0].detach().float()
            axes=(0,)+tuple(range(2,x.ndim))
            result[key]={'running_mean':module.running_mean.detach().cpu().tolist(),
                         'running_var':module.running_var.detach().cpu().tolist(),
                         'input_mean':x.mean(axes).cpu().tolist(),
                         'input_var_biased':x.var(axes,unbiased=False).cpu().tolist(),
                         'mode':'batch_statistics' if module.training else 'saved_statistics'}
        handles.append(layer.register_forward_pre_hook(capture))
    return result,handles


def candidate_invariance(model,pixels,actions):
    """Invoke the actual rollout and goal criterion through get_cost()."""
    from mylewm.tools.audit_raw_path import difference
    model.eval()
    # One context image, three future actions; four candidate plans.
    base=actions[:1,:1].expand(1,4,actions.shape[-1]).clone()
    plans=torch.stack([base,base+.05,base-.1,base+.2],dim=1)
    def cost(candidate):
        n=candidate.shape[1]
        info={'pixels':pixels[:1,:1].unsqueeze(1).expand(-1,n,-1,-1,-1,-1).clone(),
              'goal':pixels[:1,-1:].unsqueeze(1).expand(-1,n,-1,-1,-1,-1).clone(),
              'action':actions[:1,:1].unsqueeze(1).expand(-1,n,-1,-1).clone()}
        with torch.no_grad():return model.get_cost(info,candidate.clone())
    full=cost(plans)
    split=torch.cat([cost(plans[:,i:i+1]) for i in range(4)],dim=1)
    order=torch.tensor([2,0,3,1],device=plans.device)
    reordered=cost(plans[:,order])[:,torch.argsort(order)]
    extended=cost(torch.cat([plans,plans[:,:1]+.75],dim=1))[:,:4]
    return {'split':difference(full,split,atol=1e-5,rtol=1e-5),
            'reorder':difference(full,reordered,atol=1e-5,rtol=1e-5),
            'unrelated_candidate':difference(full,extended,atol=1e-5,rtol=1e-5),
            'costs':full.cpu().tolist(),'mode':'eval; actual JEPA.get_cost/rollout/criterion'}

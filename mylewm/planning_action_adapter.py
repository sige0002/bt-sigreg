"""Use a shared physical CEM search distribution with checkpoint-specific inputs."""
import torch


class PlanningActionAdapter(torch.nn.Module):
    def __init__(self,model,reference_mean,reference_std,training_mean,training_std):
        super().__init__()
        self.model=model
        device=next(model.parameters()).device
        for name,value in [('reference_mean',reference_mean),('reference_std',reference_std),
                           ('training_mean',training_mean),('training_std',training_std)]:
            self.register_buffer(name,torch.as_tensor(value,device=device,dtype=torch.float64))
        if not (self.reference_mean.shape==self.reference_std.shape==self.training_mean.shape==self.training_std.shape):
            raise ValueError('Action normalization dimensions differ')
        if any(not torch.isfinite(v).all() for v in (self.reference_mean,self.reference_std,self.training_mean,self.training_std)) or not (self.reference_std>0).all() or not (self.training_std>0).all():
            raise ValueError('Invalid action normalization')

    def normalize_for_model(self,actions):
        d=self.reference_mean.numel()
        shaped=actions.reshape(*actions.shape[:-1],-1,d)
        converted=(shaped*self.reference_std+self.reference_mean-self.training_mean)/self.training_std
        return converted.reshape_as(actions).to(actions.dtype)

    def get_cost(self,info,action_candidates):
        adjusted=dict(info)
        if 'action' in adjusted:adjusted['action']=self.normalize_for_model(adjusted['action'])
        return self.model.get_cost(adjusted,self.normalize_for_model(action_candidates))

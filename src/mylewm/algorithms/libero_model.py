"""Parameter-shared two-camera adapter, common to all regularizer conditions."""
import torch
from jepa import JEPA


class TwoViewJEPA(JEPA):
    def encode(self, info):
        pixels=info['pixels'].float()
        if pixels.ndim!=6 or pixels.shape[2]!=2:
            raise ValueError('Expected (batch,time,2,channels,height,width)')
        b,t,v,c,h,w=pixels.shape
        out=self.encoder(pixels.reshape(b*t*v,c,h,w),interpolate_pos_encoding=True)
        cls=out.last_hidden_state[:,0].reshape(b*t,v,-1).flatten(1)
        info['emb']=self.projector(cls).reshape(b,t,-1)
        if 'action' in info: info['act_emb']=self.action_encoder(info['action'])
        return info

"""Reuse fixed image embeddings within each upstream CEM candidate search."""
import torch
from stable_worldmodel.solver import CEMSolver


class CachedCEMSolver(CEMSolver):
    """Preserve upstream sampling/order; cache only action-independent images."""

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        model = getattr(self.model, 'model', self.model)
        from jepa import JEPA
        if type(model) is not JEPA:
            raise TypeError('Cached CEM requires the audited JEPA image encoding path')
        if model.training:
            raise ValueError('Image caching requires an evaluation-mode model')
        original_cost = self.model.get_cost
        original_encode = model.encode
        cost_calls = 0
        encode_slot = 0
        cache = {}

        def encode(info):
            nonlocal encode_slot
            slot = encode_slot
            encode_slot += 1
            if slot not in cache:
                result = original_encode(info)
                cache[slot] = result['emb']
                return result
            info['emb'] = cache[slot]
            if 'action' in info:
                info['act_emb'] = model.action_encoder(info['action'])
            return info

        def cost(info, candidates):
            nonlocal cost_calls, encode_slot
            # Upstream completes all iterations for one environment batch
            # before moving to the next. Never reuse across batches/replans.
            if cost_calls % self.n_steps == 0:
                cache.clear()
            cost_calls += 1
            encode_slot = 0
            # JEPA only encodes sample zero. Slice BEFORE CPU->CUDA transfer
            # to avoid materializing the 300 identical candidate images.
            info = dict(info)
            for key in ('pixels', 'goal', 'goal_pixels'):
                if key in info and torch.is_tensor(info[key]):
                    info[key] = info[key][:, :1]
            return original_cost(info, candidates)

        model.encode = encode
        self.model.get_cost = cost
        try:
            return super().solve(info_dict, init_action=init_action)
        finally:
            model.encode = original_encode
            self.model.get_cost = original_cost

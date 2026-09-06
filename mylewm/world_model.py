"""Shared E/A/F with README §7 normalization in training and CEM inference."""

from jepa import JEPA
from multitask_lewm import normalize_latent


class MyLeWM(JEPA):
    def encode(self, info):
        result = super().encode(info)
        result['emb'] = normalize_latent(result['emb'])
        return result

    def predict(self, emb, act_emb):
        return normalize_latent(super().predict(normalize_latent(emb), act_emb))

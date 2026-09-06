"""Model factory; legacy same-frame smoke trainer is deliberately disabled."""
import json
from pathlib import Path
import torch
import stable_pretraining as spt
from world_model import MyLeWM as JEPA
from module import ARPredictor, Embedder, MLP

def build_model(config_path: Path):
    cfg = json.loads(config_path.read_text()); clean=lambda d:{k:v for k,v in d.items() if not k.startswith('_')}
    enc=spt.backbone.utils.vit_hf(cfg['encoder']['size'],patch_size=cfg['encoder']['patch_size'],image_size=cfg['encoder']['image_size'],pretrained=False,use_mask_token=False)
    mlp=lambda k: MLP(input_dim=cfg[k]['input_dim'],output_dim=cfg[k]['output_dim'],hidden_dim=cfg[k]['hidden_dim'],norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=enc,predictor=ARPredictor(**clean(cfg['predictor'])),action_encoder=Embedder(**clean(cfg['action_encoder'])),projector=mlp('projector'),pred_proj=mlp('pred_proj'))

def main():
    raise RuntimeError('Retired: this smoke trainer used same-frame targets and duplicated views. Use train_crossed.py with measured correspondence tables. Old checkpoints are invalid for research comparison.')

if __name__ == '__main__': main()

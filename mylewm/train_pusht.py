"""Minimal real-data myLeWM training entrypoint for PushT.

It intentionally uses the official LeWM checkpoint/model construction and
replaces only the objective.  Start with ``--steps 1`` to verify the complete
data -> E/A/F -> loss -> optimizer path before launching a long run.
"""
import argparse, json
from pathlib import Path
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from multitask_jepa import MultiTaskJEPAObjective

def build_model(config_path: Path):
    cfg = json.loads(config_path.read_text()); clean=lambda d:{k:v for k,v in d.items() if not k.startswith('_')}
    enc=spt.backbone.utils.vit_hf(cfg['encoder']['size'],patch_size=cfg['encoder']['patch_size'],image_size=cfg['encoder']['image_size'],pretrained=False,use_mask_token=False)
    mlp=lambda k: MLP(input_dim=cfg[k]['input_dim'],output_dim=cfg[k]['output_dim'],hidden_dim=cfg[k]['hidden_dim'],norm_fn=torch.nn.BatchNorm1d)
    return JEPA(encoder=enc,predictor=ARPredictor(**clean(cfg['predictor'])),action_encoder=Embedder(**clean(cfg['action_encoder'])),projector=mlp('projector'),pred_proj=mlp('pred_proj'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=1); p.add_argument('--batch-size',type=int,default=1); p.add_argument('--lr',type=float,default=1e-5); p.add_argument('--cache-dir',default='.cache/stable-wm'); p.add_argument('--output',default='.cache/stable-wm/pusht/mylewm_object.ckpt'); args=p.parse_args()
    root=Path(args.cache_dir); model=build_model(root/'hf_pusht/config.json'); model.train(); device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)
    ds=swm.data.HDF5Dataset('pusht_expert_train',num_steps=3,keys_to_load=['pixels','action'],cache_dir=root)
    loader=torch.utils.data.DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr); objective=MultiTaskJEPAObjective(model,target_detach=True)
    for step,batch in zip(range(args.steps),loader):
        batch={k:v.to(device) for k,v in batch.items()}
        expected_action = model.action_encoder.patch_embed.in_channels
        if batch['action'].shape[-1] < expected_action:
            batch['action'] = torch.nn.functional.pad(batch['action'], (0, expected_action - batch['action'].shape[-1]))
        elif batch['action'].shape[-1] > expected_action:
            batch['action'] = batch['action'][..., :expected_action]
        opt.zero_grad(set_to_none=True)
        # Two identical views make the first real-data smoke path explicit;
        # replace view_b with an augmentation/camera stream for experiments.
        contexts={'view_a':batch,'view_b':{k:v.clone() for k,v in batch.items()}}
        futures={'view_a':{'pixels':batch['pixels']},'view_b':{'pixels':batch['pixels']}}
        loss,metrics=objective.loss_from_observations(contexts,futures,horizons=(1,))
        loss.backward(); opt.step(); print({'step':step,'loss':float(loss.detach()),'metrics':metrics})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); torch.save(model, args.output); print('saved', args.output)

if __name__ == '__main__': main()

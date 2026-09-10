"""Offline latent prediction from either storage format using training statistics.

This measures prediction error, not robot/environment success. Trusted object
checkpoints only. Default dry-run; --execute writes a new output directory.
"""
import argparse
import json
from pathlib import Path

import torch

from mylewm import train
from mylewm.data_contract import file_sha256, verify_training_data
from mylewm.input_contract import inference_manifest


def inference_dataset(model, manifest, split='test', legacy_contract=None):
    verify_training_data(manifest)
    prepared = inference_manifest(model, manifest, legacy_contract)
    ds = train.clip_dataset(prepared)
    episodes = manifest[split + '_episodes']
    if len(set(episodes)) != len(episodes) or any(type(e) is not int or not 0 <= e < len(ds.lengths) for e in episodes):
        raise ValueError('Invalid inference episode IDs')
    allowed = set(episodes)
    ids = [i for i, (ep, _) in enumerate(ds.clip_indices) if ep in allowed]
    if not ids:
        raise ValueError('Empty inference split')
    return torch.utils.data.Subset(ds, ids)


@torch.inference_mode()
def predict(model, batch):
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    batch['action'] = torch.nan_to_num(batch['action'])
    output = model.encode(batch)
    prediction = model.predict(output['emb'][:, :3], output['act_emb'][:, :3])
    target = output['emb'][:, 1:]
    return prediction, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--training-contract', type=Path,
                   help='Verified training conditions for an old checkpoint lacking a saved contract')
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    p.add_argument('--limit', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-data', action='store_true')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit < 1 or args.batch_size < 1 or not args.checkpoint.is_file():
        raise ValueError('Require a checkpoint and positive limit/batch size')
    manifest = json.loads(args.manifest.read_text())
    if not args.execute:
        print(json.dumps({'checkpoint': str(args.checkpoint), 'data_format': manifest.get('data_format', 'hdf5'),
                          'split': args.split, 'limit': args.limit, 'execute': False}))
        print('Dry-run: no checkpoint load, GPU initialization, or output creation')
        return
    torch.set_num_threads(4)
    identity = verify_training_data(manifest, full=args.verify_data)
    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False).eval()
    legacy = json.loads(args.training_contract.read_text()) if args.training_contract else None
    ds = inference_dataset(model, manifest, args.split, legacy)
    selected = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output / 'status.json'
    status.write_text(json.dumps({'state': 'running'}))
    try:
        predictions, targets = [], []
        for batch in torch.utils.data.DataLoader(selected, batch_size=args.batch_size, num_workers=0):
            pred, target = predict(model, batch)
            if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
                raise ValueError('Non-finite latent prediction')
            predictions.append(pred.cpu())
            targets.append(target.cpu())
            print(f'Predicted {sum(len(v) for v in predictions)}/{len(selected)} clips', flush=True)
        pred, target = torch.cat(predictions), torch.cat(targets)
        cases = [ds.dataset.clip_indices[i] for i in ds.indices[:len(selected)]]
        torch.save({'prediction': pred, 'target': target, 'episode_start': cases}, args.output / 'predictions.pt')
        result = {'state': 'succeeded', 'clips': len(selected), 'latent_mse': (pred - target).square().mean().item(),
                  'metric': 'offline one-step latent MSE; not control success',
                  'checkpoint_sha256': file_sha256(args.checkpoint),
                  'manifest_sha256': file_sha256(args.manifest), 'data_identity': identity,
                  'input_contract': getattr(model, 'input_contract', legacy),
                  'normalization': 'checkpoint_training_statistics', 'split': args.split}
        (args.output / 'results.json').write_text(json.dumps(result, indent=2))
        status.write_text(json.dumps({'state': 'succeeded', 'clips': len(selected)}))
    except Exception as exc:
        status.write_text(json.dumps({'state': 'failed', 'error': str(exc)}))
        raise


if __name__ == '__main__':
    main()

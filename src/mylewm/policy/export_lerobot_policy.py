"""Convert a trusted world-model object ckpt into a LeRobot policy directory."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from mylewm.data.verification import file_sha256


def export(checkpoint, output, contract, planner, camera_key='observation.image'):
    from mylewm.policy.lerobot import BTSIGRegConfig, BTSIGRegPolicy
    from mylewm.policy.lerobot.processor_bt_sigreg import make_bt_sigreg_pre_post_processors
    checkpoint, output = Path(checkpoint), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = BTSIGRegConfig(input_contract=contract, planner=planner, camera_key=camera_key, device='cpu')
    policy = BTSIGRegPolicy.from_checkpoint(checkpoint, config)
    from mylewm.policy.runtime import load_world_model
    original = load_world_model(checkpoint, legacy_contract=contract)
    import torch
    # Check behavior as well as tensor shapes: a pickle can contain different
    # parameter-free module settings which a state_dict alone cannot describe.
    probe = {'pixels': torch.linspace(-1, 1, 3 * 3 * 224 * 224).reshape(1, 3, 3, 224, 224),
             'action': torch.linspace(-1, 1, 3 * len(contract['action_names']) * contract['frameskip']).reshape(1, 3, -1)}
    with torch.inference_mode():
        old = original.encode({k: v.clone() for k, v in probe.items()})
        new = policy.model.encode({k: v.clone() for k, v in probe.items()})
        torch.testing.assert_close(old['emb'], new['emb'], rtol=0, atol=0)
        torch.testing.assert_close(original.predict(old['emb'], old['act_emb']),
                                   policy.model.predict(new['emb'], new['act_emb']), rtol=0, atol=0)
    output.mkdir(parents=True, exist_ok=False)
    policy.save_pretrained(output)
    pre, post = make_bt_sigreg_pre_post_processors(config)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    # Verify the actual saved representation, including strict architecture
    # reconstruction and training normalization buffers.
    restored = BTSIGRegPolicy.from_pretrained(output)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    receipt = {'source_checkpoint_sha256': file_sha256(checkpoint), 'architecture': config.architecture,
               'versions': {k: importlib.metadata.version(k) for k in ('lerobot', 'torch', 'transformers', 'safetensors')},
               'files': {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()},
               'verification': 'strict reload; exact weights/statistics; matching encoder/predictor probe',
               'training_action_mean': restored.model.training_action_mean.tolist(),
               'training_action_std': restored.model.training_action_std.tolist()}
    from mylewm.paths import ROOT, source_files
    root = ROOT
    paths = [*source_files(), root / 'lewm/jepa.py',
             root / 'lewm/module.py', root / 'lewm/utils.py', root / 'lewm/config/train/model/lewm.yaml']
    receipt['source_sha256'] = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    (output / 'export.json').write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--planner-config', type=Path, required=True)
    p.add_argument('--training-contract', type=Path, help='Verified conditions for legacy ckpt only')
    p.add_argument('--camera-key', default='observation.image')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.execute:
        print('Dry-run: no checkpoint load or export. Add --execute to convert.')
        return
    from mylewm.policy.runtime import load_checkpoint_controller
    import torch
    torch.set_num_threads(4)
    legacy = json.loads(args.training_contract.read_text()) if args.training_contract else None
    planner = json.loads(args.planner_config.read_text())
    controller = load_checkpoint_controller(args.checkpoint, planner, legacy_contract=legacy)
    result = export(args.checkpoint, args.output, controller.contract, planner, args.camera_key)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

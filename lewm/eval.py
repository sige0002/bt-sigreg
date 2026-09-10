import os

os.environ["MUJOCO_GL"] = "egl"

import time
import hashlib
import json
import random
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    from mylewm.pusht_eval_data import EvaluationDataset
    explicit_path = cfg.eval.get('dataset_path')
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if path.suffix != '.h5':
            raise ValueError('PushT evaluator requires a .h5 file')
        return EvaluationDataset(str(path.with_suffix('')), cache_dir=path.parent,
                                    keys_to_load=cfg.dataset.keys_to_cache)
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    # stable-worldmodel >=0.1 resolves datasets through its format registry.
    # Older LeWM revisions exposed HDF5Dataset directly; retain compatibility
    # with both APIs so the evaluation entry point remains usable.
    hdf5_cls = EvaluationDataset
    if hdf5_cls is not None:
        dataset = hdf5_cls(
            dataset_name,
            keys_to_cache=cfg.dataset.keys_to_cache,
            cache_dir=dataset_path,
        )
    else:
        dataset = swm.data.load_dataset(
            dataset_name,
            # load_dataset resolves named datasets under <cache>/datasets.
            cache_dir=dataset_path,
            keys_to_load=cfg.dataset.keys_to_cache,
        )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    identity=None
    if cfg.eval.get('audit_provenance',False):
        from mylewm.evaluation_contract import provenance
        root=Path(__file__).resolve().parents[1]
        base=Path(swm.data.utils.get_cache_dir())
        dataset_path = Path(cfg.eval.dataset_path) if cfg.eval.get('dataset_path') else base/'datasets'/f'{cfg.eval.dataset_name}.h5'
        full = cfg.eval.get('verify_data', False)
        print('Evaluation preflight: ' + ('full dataset SHA-256 scan' if full else 'metadata only; no full dataset scan'), flush=True)
        identity=provenance(dataset_path,
            Path(cfg.eval.manifest),base/(cfg.policy+'_object.ckpt'),root,verify_data=full)
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    print('Loading HDF5 and action statistics', flush=True)
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        local_object = Path(swm.data.utils.get_cache_dir(), cfg.policy + "_object.ckpt")
        try:
            model = torch.load(local_object, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False) if local_object.exists() else swm.wm.utils.load_pretrained(cfg.policy)
        except (AttributeError, FileNotFoundError):
            # stable-worldmodel 0.0.x has no wm.utils; load the serialized
            # official LeWM object produced from the HF weights.
            model = torch.load(
                Path(swm.data.utils.get_cache_dir(), cfg.policy + "_object.ckpt"),
                map_location="cuda" if torch.cuda.is_available() else "cpu",
                weights_only=False,
            )
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        # Proposed models persist their training preprocessing contract.
        # Official checkpoints without these buffers keep the original path.
        if hasattr(model, 'training_action_mean'):
            mean = model.training_action_mean.detach().cpu().numpy()
            std = model.training_action_std.detach().cpu().numpy()
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
                raise ValueError('Invalid action statistics in checkpoint')
            if mean.shape != process['action'].mean_.shape or std.shape != mean.shape:
                raise ValueError('Checkpoint action statistics do not match environment')
            if cfg.eval.get('shared_physical_search',False):
                from mylewm.planning_action_adapter import PlanningActionAdapter
                model=PlanningActionAdapter(model,process['action'].mean_,process['action'].scale_,mean,std).eval()
                print('action_statistics_source: checkpoint via common physical CEM search')
            else:
                process['action'].mean_ = mean
                process['action'].scale_ = std
                process['action'].var_ = std ** 2
                print('action_statistics_source: checkpoint')
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    # Sample one valid start row per unique episode.  Sampling rows directly
    # can select the same episode multiple times and is not an independent
    # episode evaluation.
    valid_episode_ids = np.unique(dataset.get_col_data(col_name)[valid_indices])
    if len(valid_episode_ids) < cfg.eval.num_eval:
        raise ValueError("Not enough unique episodes for evaluation.")
    selected_episode_ids = np.sort(g.choice(valid_episode_ids, size=cfg.eval.num_eval, replace=False))
    selected_rows = []
    for ep_id in selected_episode_ids:
        rows = valid_indices[dataset.get_col_data(col_name)[valid_indices] == ep_id]
        selected_rows.append(int(g.choice(rows)))
    random_episode_indices = np.array(sorted(selected_rows), dtype=int)

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if cfg.eval.get('manifest'):
        manifest = json.loads(Path(cfg.eval.manifest).read_text())
        partition = cfg.eval.get('partition', 'pilot')
        offset = cfg.eval.get('offset', 0)
        cases = manifest[partition][offset:offset+cfg.eval.num_eval]
        if len(cases) != cfg.eval.num_eval:
            raise ValueError('Manifest does not contain requested evaluation cases')
        eval_episodes = np.asarray([case['episode'] for case in cases])
        eval_start_idx = np.asarray([case['start'] for case in cases])
        if len(np.unique(eval_episodes)) != cfg.eval.num_eval:
            raise ValueError('Evaluation cases must use unique episodes')

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    physical_actions=[]; initial_runtime_hashes=[]
    if identity is not None:
        from mylewm.evaluation_contract import array_hash
        original_get_actions=world._get_actions
        def audited_get_actions():
            print(f'CEM planning: environment step calls={len(physical_actions)}, cases={cfg.eval.num_eval}', flush=True)
            if not initial_runtime_hashes:
                for index in range(cfg.eval.num_eval):
                    initial_runtime_hashes.append({key:array_hash(value[index]) for key,value in world.infos.items()
                        if key in ('pixels','state','proprio','goal','goal_pixels','goal_state','goal_proprio')})
            return original_get_actions()
        world._get_actions=audited_get_actions
        original_env_step=world.envs.step
        def audited_env_step(actions,mask=None):
            physical_actions.append({'actions':np.asarray(actions).tolist(),
                'mask':np.ones(cfg.eval.num_eval,dtype=bool).tolist() if mask is None else np.asarray(mask).tolist()})
            result = original_env_step(actions,mask=mask)
            print(f'Environment step call {len(physical_actions)} completed; active cases={cfg.eval.num_eval if mask is None else int(np.asarray(mask).sum())}', flush=True)
            return result
        world.envs.step=audited_env_step

    results_path.mkdir(parents=True, exist_ok=True)
    if identity is not None and (results_path/(cfg.output.filename+'.json')).exists():
        raise FileExistsError('Audited evaluation result already exists')

    start_time = time.time()
    print(f'Starting evaluation: {cfg.eval.num_eval} cases, budget={cfg.eval.eval_budget} per case', flush=True)
    # Seed environment RNGs before the dataset evaluator's reset(seed=None).
    world.reset(seed=cfg.seed)
    # The upstream evaluation API changed after the HDF5 release.  Use the
    # replay evaluator when available; otherwise run the current environment
    # evaluator with the same episode count and budget.
    import inspect
    if "dataset" in inspect.signature(world.evaluate).parameters:
        metrics = world.evaluate(
            dataset=dataset, start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=results_path,
        )
    else:
        raise RuntimeError("Dataset-driven evaluation API is unavailable; refusing random fallback")

    ckpt = Path(swm.data.utils.get_cache_dir(), cfg.policy + "_object.ckpt")
    if ckpt.exists():
        sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        print(f"checkpoint: {ckpt} sha256={sha}")
    print(f"evaluated_unique_episodes: {len(np.unique(eval_episodes))}")
    print(f"start_steps: {eval_start_idx.tolist()}")
    end_time = time.time()
    
    print(metrics)

    machine_result = {
        'policy':cfg.policy, 'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest() if ckpt.exists() else None,
        'episodes':eval_episodes.tolist(), 'starts':eval_start_idx.tolist(),
        'successes':np.asarray(metrics['episode_successes'],dtype=bool).tolist(),
        'success_rate':float(metrics['success_rate']),
        'config':OmegaConf.to_container(cfg,resolve=True), 'elapsed':end_time-start_time,
    }
    if identity is not None:
        machine_result.update({'provenance':identity,'physical_actions':physical_actions,
            'initial_runtime_hashes':initial_runtime_hashes,
            'planner_action_mean':process['action'].mean_.tolist(),
            'planner_action_std':process['action'].scale_.tolist(),
            'shared_physical_search':bool(cfg.eval.get('shared_physical_search',False))})
    (results_path / (cfg.output.filename + '.json')).write_text(json.dumps(machine_result,indent=2))

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()

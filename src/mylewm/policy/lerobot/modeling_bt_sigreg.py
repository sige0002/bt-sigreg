import copy

import torch
from lerobot.policies.pretrained import PreTrainedPolicy

from mylewm.data.contract import model_contract
from mylewm.policy.runtime import WorldModelController, load_world_model
from .configuration_bt_sigreg import BTSIGRegConfig


class BTSIGRegPolicy(PreTrainedPolicy):
    config_class = BTSIGRegConfig
    name = 'bt_sigreg'

    def __init__(self, config, **kwargs):
        super().__init__(config)
        config.validate_features()
        from mylewm.training.train import build_model
        dim = len(config.input_contract['action_names'])
        # Preserve caller RNG; reconstruction must not alter application RNG.
        with torch.random.fork_rng(devices=[]):
            self.model = build_model(dim * config.input_contract['frameskip'])
        self.model.input_contract = copy.deepcopy(config.input_contract)
        self.model.register_buffer('training_action_mean', torch.zeros(dim, dtype=torch.float64))
        self.model.register_buffer('training_action_std', torch.ones(dim, dtype=torch.float64))
        self.controller = None

    @classmethod
    def from_checkpoint(cls, path, config):
        model = load_world_model(path, legacy_contract=config.input_contract)
        if model_contract(model, config.input_contract) != config.input_contract:
            raise ValueError('Checkpoint input contract mismatch')
        policy = cls(config)
        # Strict keys/shapes reject unsupported architectures, including LIBERO
        # multi-camera models. Keep the reconstructed model, never pickle it.
        policy.model.load_state_dict(model.state_dict(), strict=True)
        policy.to(config.device).eval()
        policy.reset()
        return policy

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        if kwargs.get('strict', True) is not True:
            raise ValueError('BT-SIGReg requires strict weight loading')
        kwargs['strict'] = True
        # Planning budgets may change explicitly; the trained input semantics
        # must not be replaced by a same-shaped but incompatible config.
        from lerobot.configs.policies import PreTrainedConfig
        saved = PreTrainedConfig.from_pretrained(pretrained_name_or_path,
            **{k: v for k, v in kwargs.items() if k in ('revision', 'cache_dir', 'local_files_only', 'token')})
        if not isinstance(saved, BTSIGRegConfig):
            raise ValueError('Expected a BT-SIGReg policy config')
        configured = kwargs.get('config')
        if configured is not None and (configured.input_contract != saved.input_contract or configured.architecture != saved.architecture):
            raise ValueError('Cannot override trained input contract or architecture')
        if configured is None:
            kwargs['config'] = saved
        policy = super().from_pretrained(pretrained_name_or_path, **kwargs)
        policy.reset()
        return policy

    def reset(self):
        self.controller = WorldModelController(self.model, self.config.planner)

    def prime(self, images, past_actions, goal, timestamps):
        self.reset()
        self.controller.prime(images, past_actions, goal, timestamps)

    @torch.inference_mode()
    def select_action(self, batch, **kwargs):
        if self.controller is None:
            raise ValueError('Call prime with actual history first')
        image = batch[self.config.camera_key]
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError('This policy controls one robot (batch size 1)')
        executed = batch.get('observation.executed_action')
        if executed is not None:
            executed = torch.as_tensor(executed)
            dim = len(self.config.input_contract['action_names'])
            if executed.shape == (1, dim):
                executed = executed[0]
            elif executed.shape != (dim,):
                raise ValueError('executed_action must be (action_dim,) or (1, action_dim)')
        return self.controller.select_action(image, float(batch['observation.timestamp']), executed)[None]

    @torch.inference_mode()
    def predict_action_chunk(self, batch, **kwargs):
        # Explicit full-history API for applications that manage their own
        # action execution/history. Does not change the select_action queue.
        runtime = WorldModelController(self.model, self.config.planner)
        runtime.prime(batch['history'], batch['past_actions'], batch['goal'], batch['timestamps'])
        return runtime.plan()[None]

    def get_optim_params(self):
        raise NotImplementedError('Use mylewm.training.train for world-model training')

    def forward(self, batch):
        raise NotImplementedError('This is an inference policy; use mylewm.training.train for training')

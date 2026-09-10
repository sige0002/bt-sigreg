from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

from mylewm.data.contract import validate_contract
from mylewm.policy.runtime import validate_planner


@PreTrainedConfig.register_subclass('bt_sigreg')
@dataclass
class BTSIGRegConfig(PreTrainedConfig):
    architecture: str = 'lewm_tiny_v1'
    input_contract: dict = field(default_factory=dict)
    planner: dict = field(default_factory=dict)
    camera_key: str = 'observation.image'
    n_obs_steps: int = 3
    push_to_hub: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.architecture != 'lewm_tiny_v1' or self.n_obs_steps != 3:
            raise ValueError('Only lewm_tiny_v1 with history=3 is supported')
        self.input_contract = validate_contract(self.input_contract)
        self.planner = validate_planner(self.planner, len(self.input_contract['action_names']))
        if not self.camera_key.startswith('observation.image'):
            raise ValueError('Use a LeRobot observation.image(s) camera key')
        if not self.input_features:
            self.input_features = {self.camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))}
        if not self.output_features:
            self.output_features = {'action': PolicyFeature(type=FeatureType.ACTION, shape=(len(self.input_contract['action_names']),))}
        self.validate_features()

    def validate_features(self):
        expected = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224))
        if self.input_features != {self.camera_key: expected}:
            raise ValueError('Require one RGB camera with the model preprocessing size 224')
        if self.output_features != {'action': PolicyFeature(type=FeatureType.ACTION, shape=(len(self.input_contract['action_names']),))}:
            raise ValueError('Action features disagree with input contract')

    @property
    def observation_delta_indices(self):
        k = self.input_contract['frameskip']
        return [-2 * k, -k, 0]

    @property
    def action_delta_indices(self):
        return list(range(self.input_contract['frameskip']))

    @property
    def reward_delta_indices(self):
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError('Train the world model with mylewm.training.train, then export the policy')

    def get_scheduler_preset(self):
        return None

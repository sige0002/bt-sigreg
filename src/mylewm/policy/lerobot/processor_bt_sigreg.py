"""Batch/device adapters; image/action normalization lives in shared runtime."""
from lerobot.processor import AddBatchDimensionProcessorStep, DeviceProcessorStep, PolicyProcessorPipeline
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME


def make_bt_sigreg_pre_post_processors(config, dataset_stats=None):
    # Never fit or substitute evaluation dataset statistics. The exported model
    # carries the training buffers. Outputs already use physical coordinates.
    return (
        PolicyProcessorPipeline(steps=[AddBatchDimensionProcessorStep(), DeviceProcessorStep(device=config.device)],
                                name=POLICY_PREPROCESSOR_DEFAULT_NAME),
        PolicyProcessorPipeline(steps=[DeviceProcessorStep(device='cpu')],
                                name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
                                to_transition=policy_action_to_transition, to_output=transition_to_policy_action),
    )

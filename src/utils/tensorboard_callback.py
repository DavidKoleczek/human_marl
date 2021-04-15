''' Callback for evaulating HITL training in Tensorboard with Stable Baselines 3
'''

from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('rollout/interventions', self.model.get_env().get_attr('interventions_made')[0])
        self.logger.record('rollout/raw_reward', self.model.get_env().get_attr('raw_reward')[0])
        self.logger.record('rollout/modified_reward', self.model.get_env().get_attr('modified_reward')[0])
        try:
            intervention_rate = self.model.get_env().get_attr('interventions_made')[0] / self.model.get_env().get_attr('timesteps')[0]
        except ZeroDivisionError:
            intervention_rate = 0
        self.logger.record('rollout/intervention_rate', intervention_rate)

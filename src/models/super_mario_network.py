import torch
from all.approximation import Approximation
from all.core import State
import torch.nn as nn

class QNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            name='q',
            **kwargs
    ):
        model = QModule(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class RLNetwork(nn.Module):
    """
    Wraps a network such that States can be given as input.
    """
    def __init__(self, model, _=None):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, state):
        return state.apply(self.model, 'observation', 'pilot_action')

class QModule(RLNetwork):
    def forward(self, states, actions=None):
        values = super().forward(states)
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return values.gather(1, actions.view(-1, 1)).squeeze(1)



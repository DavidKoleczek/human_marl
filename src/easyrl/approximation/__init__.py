from .approximation import Approximation
from .architectures import RLNetwork
from .v_network import VNetwork
from .target import TargetNetwork, TrivialTarget, PolyakTarget
from .feature_network import FeatureNetwork
from .q_continuous import QContinuous
from .layers import fc_q, fc_v, fc_soft_policy, fc_actor_critic
from .sac_discrete_approx import fc_q_discrete, fc_v_discrete, fc_soft_policy_discrete

__all__ = ['Approximation',
           'RLNetwork',
           'VNetwork',
           'TargetNetwork',
           'TrivialTarget',
           'PolyakTarget',
           'FeatureNetwork',
           'QContinuous',
           'fc_q',
           'fc_v',
           'fc_soft_policy',
           'fc_q_discrete',
           'fc_v_discrete',
           'fc_soft_policy_discrete',
           'fc_actor_critic']

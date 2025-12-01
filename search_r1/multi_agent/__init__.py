"""
Multi-agent search and reasoning framework for CoSearch-R1.
"""

from .protocol import MultiAgentProtocol, AgentRole
from .templates import MultiAgentTemplate
from .rollout import MultiAgentRolloutManager

__all__ = [
    'MultiAgentProtocol',
    'AgentRole',
    'MultiAgentTemplate',
    'MultiAgentRolloutManager',
]


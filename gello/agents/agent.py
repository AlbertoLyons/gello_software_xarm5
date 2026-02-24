# Importación de liberías necesarias
from typing import Any, Dict, Protocol
import numpy as np

"""
agent.py
Script que define la clase base Agent y un dummyAgent.
"""
"""
Clase agente que define la interfaz para los agentes que interactúan con el entorno.
Parametros:
    obs: observación del entorno.
"""
class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

"""
Clase DummyAgent que implementa la interfaz Agent y devuelve una acción de ceros.
Parametros:
    num_dofs: número de grados de libertad del robot.
    obs: observación del entorno.
"""
class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs
    # Función que devuelve una acción de ceros para cada grado de libertad
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(self.num_dofs)
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            assert L == half_dim * 2, f"{key} must be even, something is wrong"
            left_obs[key] = val[:half_dim]
            right_obs[key] = val[half_dim:]
        return np.concatenate(
            [self.agent_left.act(left_obs), self.agent_right.act(right_obs)]
        )

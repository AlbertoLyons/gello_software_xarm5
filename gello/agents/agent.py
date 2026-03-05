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

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        # Simplemente devolvemos ceros para las articulaciones que tenemos (5 en tu caso)
        return np.zeros(self.num_dofs)
        

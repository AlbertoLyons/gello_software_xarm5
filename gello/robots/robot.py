# Importación de librerías necesarias
from abc import abstractmethod
from typing import Dict, Protocol
import numpy as np

"""
robot.py
Script que define la clase Robot, que es una interfaz para controlar un robot, ya sea simulado o físico.
"""
"""
Clase Robot, que es un protocolo para controlar un robot.
"""
class Robot(Protocol):
    # Obtiene el número de grados de libertad del robot, y lo devuelve
    @abstractmethod
    def num_dofs(self) -> int:
        raise NotImplementedError
    # Obtiene el estado actual del robot, y lo devuelve
    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        raise NotImplementedError
    # Comanda a un joint del robot a un estado dado con parametro de entrada el estado deseado del joint
    @abstractmethod
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        raise NotImplementedError
    """
    Obtiene las observaciones actuales del robot.
    Esto es para extraer toda la información que está disponible del robot, 
    como las posiciones de los joints, las velocidades de los joints, etc. 
    Esto también puede incluir información de sensores adicionales, como cámaras, sensores de fuerza, etc.
    Returns:
        Dict[str, np.ndarray]: Un diccionario de observaciones.
    """
    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

"""
Clase PrintRobot, que es un robot que imprime el estado del joint comandado. (para pruebas)
"""
class PrintRobot(Robot):
    # Inicializa el robot con el número de grados de libertad, y un flag para no imprimir el estado del joint comandado
    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        self._joint_state = np.zeros(num_dofs)
        self._dont_print = dont_print
    # Devuelve el número de grados de libertad del robot
    def num_dofs(self) -> int:
        return self._num_dofs
    # Devuelve el estado actual del robot
    def get_joint_state(self) -> np.ndarray:
        return self._joint_state
    # Comanda a un joint del robot a un estado dado con parametro de entrada el estado deseado del joint, y lo imprime
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, "
            f"got {len(joint_state)}."
        )
        self._joint_state = joint_state
        if not self._dont_print:
            print(self._joint_state)
    # Obtiene las observaciones de los joints, que incluyen la posicion, velocidad, posicion y orientacion del end effector, y la posicion del gripper
    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_state = self.get_joint_state()
        pos_quat = np.zeros(7)
        return {
            "joint_positions": joint_state,
            "joint_velocities": joint_state,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(0),
        }

def main():
    pass

if __name__ == "__main__":
    main()

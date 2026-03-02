# Importación de librerías necesarias
import time
from typing import Any, Dict, Optional
import numpy as np
# Importación de módulos de la carpeta gello
from gello.cameras.camera import CameraDriver
from gello.robots.robot import Robot
"""
env.py
Script que define la clase RobotEnv, que representa un entorno de robot con cámaras y un robot.
"""
"""
Clase Rate
La clase Rate se utiliza para controlar la frecuencia de actualización del entorno. 
"""
class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate
    # Se encarga de hacer una pausa en la ejecución del programa para mantener la frecuencia de actualización deseada.
    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()

"""
Clase RobotEnv
La clase RobotEnv representa un entorno de robot que incluye un robot y cámaras.
Args:
    - robot: un objeto de la clase Robot que representa el robot en el entorno.
    - control_rate_hz: la frecuencia de actualización del entorno en Hz (por defecto es
    - camera_dict: un diccionario opcional que mapea nombres de cámaras a objetos de la clase CameraDriver).
"""
class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
    # Obtiene el objeto del robot y lo regresa
    def robot(self) -> Robot:
        return self._robot
    # Función dummy
    def __len__(self):
        return 0
    """
    Función que avanza el entorno del robot
    Args:
        - joints: comandos de los ángulos joints para avanzar con el entorno
    Returns:
        - obs: Observación desde el entorno
    """
    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        # Asegura que el número de comandos de joints coincida con el número de grados de libertad del robot
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()
    # Obtiene la observación del entorno y lo regresa
    def get_obs(self) -> Dict[str, Any]:
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        # TODO ojo
        #observations["gripper_position"] = robot_obs["gripper_position"]
        return observations

def main() -> None:
    pass


if __name__ == "__main__":
    main()

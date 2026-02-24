# Importación de bibliotecas necesarias
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
# Importación de módulos de GELLO
from gello.agents.agent import Agent
from gello.robots.dynamixel import DynamixelRobot

"""
gello_agent.py
Script que define la clase GelloAgent, un agente que se conecta a un robot Dynamixel y obtiene su estado articular.
La clase GelloAgent utiliza una configuración de robot específica para inicializar el robot Dynamixel y proporciona un método act que devuelve el estado articular del robot.
"""
@dataclass
class DynamixelRobotConfig:
    # Son los ID de las articulaciones del robot GELLO (sin incluir el gripper). Usualmente (1, 2, 3 ...).
    joint_ids: Sequence[int]
    # Son los offsets articulares del robot GELLO. Estos son valores que se suman a las posiciones articulares leídas del robot para obtener las posiciones articulares reales.
    joint_offsets: Sequence[float]
    # Son los signos articulares del robot GELLO.
    joint_signs: Sequence[int]
    # Configuración del gripper de GELLO. Es una tupla de (gripper_joint_id, grados en posición abierta, grados en posición cerrada). Si el robot no tiene gripper, este valor es None.
    gripper_config: Tuple[int, int, int]
    # Función que se llama después de la inicialización del dataclass para verificar que las longitudes de las listas de configuración sean consistentes.
    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)
    """
    Función que crea una instancia de DynamixelRobot utilizando la configuración proporcionada en el dataclass.
    Toma como argumentos el puerto de conexión y las posiciones articulares iniciales (opcional) y devuelve un objeto DynamixelRobot configurado.
    """
    def make_robot(
        self, port: str = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AF6000708-if00", start_joints: Optional[np.ndarray] = None
    ) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )

"""
Configuración de los robots Dynamixel disponibles en el sistema. 
Es un diccionario que mapea los puertos de conexión a las configuraciones de robot correspondientes.
Se debe de colocar manualmente los joint_offsets (en radianes) correspondiente al robot teleoperadio. 
Estos valores se pueden obtener dejando el brazo teleoperado y de teleoperación en una misma posición, y luego ejecutando el script scripts/gello_get_offsets.py
Para poder encontrar los joint signs, estos se deben de medir manualmente, observando en que sentido se mueve cada joint al moverlo manualmente.
"""
PORT_CONFIG_MAP: Dict[str, DynamixelRobotConfig] = {
    # xArm5
    "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AF6000708-if00": DynamixelRobotConfig(
        joint_ids=(1, 2, 3, 4, 5),
        joint_offsets=(
            7.854,
            3.142,
            -1.571,
            3.142,
            3.142,
            
        ),
        joint_signs=(1, 1, -1, 1, 1),
        gripper_config=None,
    ),
}
"""
Clase GelloAgent que hereda de la clase Agent. 
Esta clase se encarga de inicializar un robot Dynamixel utilizando la configuración proporcionada y de implementar el método act para obtener el estado articular del robot.
"""
class GelloAgent(Agent):
    # Inicia el agente con la configuración del robo Dynamixel, y los joints iniciales.
    def __init__(
        self,
        port: str,
        dynamixel_config: Optional[DynamixelRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        # Asegura que los start_joints sean un array de numpy si se proporcionan como una lista u otro tipo de secuencia.
        if start_joints is not None and not isinstance(start_joints, np.ndarray):
            start_joints = np.array(start_joints)
        # En caso de que no exista configuración de robot proporcionada:
        if dynamixel_config is not None:
            """ 
            Se verifica que el puerto exista y esté en el mapa de configuración,
            Luego se crea el robot utilizando la configuración correspondiente al puerto.
            """
            self._robot = dynamixel_config.make_robot(
                port=port, start_joints=start_joints
            )
        # En caso contrario, se avisa que el puerto debe existir y estar en el mapa de configuración, y se crea el robot utilizando la configuración correspondiente al puerto.
        else:
            assert os.path.exists(port), port
            assert port in PORT_CONFIG_MAP, f"Port {port} not in config map"

            config = PORT_CONFIG_MAP[port]
            self._robot = config.make_robot(port=port, start_joints=start_joints)
    # Función que actualiza el estado del agente.
    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self._robot.get_joint_state()

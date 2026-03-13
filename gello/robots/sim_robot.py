# Importación de bibliotecas necesarias
import pickle
import threading
import time
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
import zmq
from dm_control import mjcf
# Importación de la clase base Robot para definir la interfaz del robot simulado
from gello.robots.robot import Robot

# Verificación de la disponibilidad del visualizador de MuJoCo
assert mujoco.viewer is mujoco.viewer

"""
sim_robot.py
Script que define un servidor de simulación utilizando MuJoCo y comunicación ZMQ.
Permite cargar modelos MJCF de brazos robóticos y grippers, simular su física en tiempo real 
y exponer una interfaz de control y lectura de sensores a través de una red local.
"""
"""
Función de utilidad para adjuntar un modelo de mano/gripper a un brazo robótico.
"""
def attach_hand_to_arm(
    arm_mjcf: mjcf.RootElement,
    hand_mjcf: mjcf.RootElement,
) -> None:

    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("attachment_site not found in arm MJCF model")

    # Expande los keyframes de 'home' para incluir los DoFs de la mano.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)

"""
Funcion que onstruye la escena de simulación (arena) cargando el brazo y opcionalmente el gripper.
Devuelve el elemento raíz MJCF con los modelos combinados.
"""
def build_scene(robot_xml_path: str, gripper_xml_path: Optional[str] = None):

    arena = mjcf.RootElement()
    arm_simulate = mjcf.from_path(robot_xml_path)

    if gripper_xml_path is not None:
        gripper_simulate = mjcf.from_path(gripper_xml_path)
        attach_hand_to_arm(arm_simulate, gripper_simulate)

    arena.worldbody.attach(arm_simulate)
    return arena

"""
Clase que es un hilo dedicado para ejecutar el servidor ZMQ sin bloquear el bucle de física de MuJoCo.
"""
class ZMQServerThread(threading.Thread):
    def __init__(self, server):
        super().__init__()
        self._server = server

    def run(self):
        self._server.serve()

    def terminate(self):
        self._server.stop()

"""
Servidor de comunicación basado en ZMQ utilizando el patrón REP (Reply).
Serializa y deserializa peticiones utilizando pickle para interactuar con la instancia del robot.
"""
class ZMQRobotServer:
    def __init__(self, robot: Robot, host: str = "127.0.0.1", port: int = 5556):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

    """Bucle principal de escucha de comandos ZMQ."""
    def serve(self) -> None:
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv()
                request = pickle.loads(message)

                method = request.get("method")
                args = request.get("args", {})
                result: Any

                # Enrutamiento de métodos hacia la instancia del robot
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    result = self._robot.get_joint_state()
                elif method == "command_joint_state":
                    result = self._robot.command_joint_state(**args)
                elif method == "get_observations":
                    result = self._robot.get_observations()
                else:
                    result = {"error": "Invalid method"}
                    raise NotImplementedError(f"Mthod {method} not implemented.")

                self._socket.send(pickle.dumps(result))
            except zmq.error.Again:
                pass # Timeout de recepción, continúa el bucle

    def stop(self) -> None:
        self._stop_event.set()
        self._socket.close()
        self._context.term()
        
"""
Clase principal que integra la simulación de MuJoCo con el servidor ZMQ.
Mantiene el estado de la física, gestiona los comandos de los actuadores
y publica las observaciones sensoriales (posiciones, velocidades, pose del EE).
"""
class MujocoRobotServer:

    def __init__(
        self,
        xml_path: str,
        gripper_xml_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
    ):
        #self._has_gripper = gripper_xml_path is not None
        arena = build_scene(xml_path, gripper_xml_path)
        self._has_gripper = "gripper" in [a.name for a in arena.actuator.all_children()]

        # Extracción de assets (mallas/meshes) para cargar el modelo en MuJoCo
        assets: Dict[str, str] = {}
        for asset in arena.asset.all_children():
            if asset.tag == "mesh":
                f = asset.file
                assets[f.get_vfs_filename()] = asset.file.contents

        xml_string = arena.to_xml_string()
        
        # Inicialización de la física de MuJoCo
        self._model = mujoco.MjModel.from_xml_string(xml_string, assets)
        self._data = mujoco.MjData(self._model)

        self._num_joints = self._model.nu
        self._joint_state = np.zeros(self._num_joints)
        self._joint_cmd = self._joint_state

        # Inicialización del servidor de comunicación
        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)
        self._print_joints = print_joints

    def num_dofs(self) -> int:
        return self._num_joints

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    # Recibe y valida los comandos de articulaciones para aplicarlos en la simulación.
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == self._num_joints, f"Expected {self._num_joints} joints."
        
        if self._has_gripper:
            _joint_state = joint_state.copy()
            # Escalamiento específico para el motor del gripper si está presente
            _joint_state[-1] = _joint_state[-1] * 255
            self._joint_cmd = _joint_state
        else:
            self._joint_cmd = joint_state.copy()

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    # Extrae información sensorial de la simulación (odometría articular y pose cartesiana).
    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_positions = self._data.qpos.copy()[: self._num_joints]
        joint_velocities = self._data.qvel.copy()[: self._num_joints]
        ee_site = "attachment_site"
        
        try:
            # Intenta obtener la posición y orientación global del efector final
            ee_pos = self._data.site_xpos.copy()[mujoco.mj_name2id(self._model, 6, ee_site)]
            ee_mat = self._data.site_xmat.copy()[mujoco.mj_name2id(self._model, 6, ee_site)]
            ee_quat = np.zeros(4)
            mujoco.mju_mat2Quat(ee_quat, ee_mat)
        except Exception:
            ee_pos = np.zeros(3)
            ee_quat = np.zeros(4)
            ee_quat[0] = 1

        gripper_pos = self._data.qpos.copy()[self._num_joints - 1]
        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": np.concatenate([ee_pos, ee_quat]),
            "gripper_position": gripper_pos,
        }

    # Inicia el hilo ZMQ y el bucle de renderizado/física pasivo de MuJoCo.
    def serve(self) -> None:
        self._zmq_server_thread.start()
        try:
            with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
                while viewer.is_running():

                    message = f"\rSim time: {self._data.time:.2f} s. Press Ctrl+C to exit."
                    print(message, end="", flush=True)

                    step_start = time.time()

                    # Aplicación de control y paso de física
                    self._data.ctrl[:] = self._joint_cmd
                    mujoco.mj_step(self._model, self._data)
                    self._joint_state = self._data.qpos.copy()[: self._num_joints]

                    if self._print_joints:
                        print(self._joint_state)

                    # Sincronización con el visualizador
                    with viewer.lock():
                        # Alterna la visualización de puntos de contacto cada 2 segundos
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self._data.time % 2)

                    viewer.sync()

                    # Control de frecuencia de la simulación para mantener tiempo real aproximado
                    time_until_next_step = self._model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            print("\nShutting down MujocoRobotServer...")
        finally:
            import os
            print("Stopping ZMQ server...")
            os._exit(0)

    def stop(self) -> None:
        self._zmq_server_thread.join()

    def __del__(self) -> None:
        self.stop()
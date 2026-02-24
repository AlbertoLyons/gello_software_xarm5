# Importación de librerías necesarias
import pickle
import threading
from typing import Any, Dict
import numpy as np
import zmq
# Importación de la clase Robot desde la carpeta GELLO
from gello.robots.robot import Robot
# Puerto por defecto para el servidor ZMQ del robot
DEFAULT_ROBOT_PORT = 6000

"""
Clase ZMQServerRobot
Esta clase representa un servidor ZMQ para el robot seguidor. 
Permite servir el estado del robot a través de ZMQ y manejar solicitudes de clientes.
Args:
    - robot: El robot seguidor que se va a servir.
    - port: El puerto en el que el servidor ZMQ escuchará las solicitudes
    - host: La dirección IP en la que el servidor ZMQ escuchará las solicitudes
"""
class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()
    # Servir el estado del robot a través de ZMQ
    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1000) # Establece tiempo de espera a 1000 ms
        while not self._stop_event.is_set():
            try:
                # Espera la siguiente solicitud del cliente
                message = self._socket.recv()
                request = pickle.loads(message)
                # Llama el método apropiado según la solicitud
                method = request.get("method")
                args = request.get("args", {})
                result: Any
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
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                # Ocurrió un Timeout. No se debe de interactuar con la consola, o se debe de terminarlo
                pass
    # Señal para detener el servidor
    def stop(self) -> None:
        self._stop_event.set()
"""
Clase ZMQClientRobot
Representa un ZMQ cliente para el robot líder.
"""
class ZMQClientRobot(Robot):

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")
    # Obtiene el número de joints y lo regresa
    def num_dofs(self) -> int:
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
    # Obtiene el estado actual del líder y lo regresa
    def get_joint_state(self) -> np.ndarray:
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")
    # Comanda el líder robot hacía el estado dado
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state},
        }
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
    # Obtiene la observación actual del robot líder y lo regresa
    def get_observations(self) -> Dict[str, np.ndarray]:
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        try:
            self._socket.send(send_message)
            result = pickle.loads(self._socket.recv())
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            return result
        except zmq.Again:
            raise RuntimeError("ZMQ timeout - robot may be disconnected")
    # Cierra el socket ZMQ y el contexto
    def close(self) -> None:
        self._socket.close()
        self._context.term()

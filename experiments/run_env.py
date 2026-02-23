# Importación de liberías necesarias
import glob
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import numpy as np
import tyro
# Importación de módulos para el funcionamiento del entorno y los agentes
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.utils.launch_utils import instantiate_from_dict
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera

"""
run_env.py

El script permite la inicialización del agente GELLO y la conexión con el entorno de control del robot.
El agente puede ser configurado para utilizar el brazo líder, o uno de prueba (dummy) para verificar la conexión y el funcionamiento del entorno.
Parametros asignables desde consola:
- agent: Define el tipo de agente a utilizar (gello, dummy, policy). Policy no implementado en este script, pero se puede agregar para aprendizaje por imitación.
- robot_port: Puerto para la comunicación con el nodo del robot  (opcional).
- wrist_camera_port: Puerto para la cámara de la muñeca (opcional).
- base_camera_port: Puerto para la cámara de base (opcional).
- hostname: Dirección IP del host para la comunicación con los nodos (opcional).
- hz: Frecuencia de control en Hz (opcional).
- start_joints: Posición inicial de las articulaciones del robot (opcional).
- gello_port: Puerto para la conexión con el agente GELLO (opcional).
- mock: Si se activa, se comprueba solamente la conexión con el brazo GELLO (opcional).
- use_save_interface: Si se activa, se activa el guardado de datos para el entrenamento 
  (s para guardar, q para terminar el guardado)
- data_dir: Directorio donde se guardarán los datos (opcional).
- use_cameras: Si se activa, se inicializan los clientes de las cámaras para propósitos de aprendizaje por imitación (opcional).
"""
@dataclass
class Args:
    agent: str = "gello"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: Path = (Path(__file__).parent.parent / "gello/data")    
    use_cameras: bool = False
    # Método que convierte los start_joints a un array de numpy si se proporciona en los args.
    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)

"""
Función principal que inicializa el entorno y el agente, y ejecuta el bucle de control.
"""
def main(args):
    # Si se activa el modo mock, se utiliza un robot cliente para comprobar la conexión con el robot GELLO
    if args.mock:
        robot_client = PrintRobot(5, dont_print=True)
        camera_clients = {}
    # Se inicializan los clientes para el robot y las cámaras utilizando ZMQ para la comunicación.
    else:
        if args.use_cameras:
            camera_clients = {
                # Añadido de nodos de cámaras para propósitos de aprendizaje por imitación
                "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
                # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
            }
        else:
            camera_clients = {}
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    # Se crea el entorno del robot utilizando el cliente del robot y los clientes de las cámaras (si se activa), con la frecuencia de control especificada.
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)
    # Se inicializa el agente
    agent_cfg = {}
    # Si se elige el agente gello (seleccionado de manera predeterminada):
    if args.agent == "gello":
        # Se busca el puerto para la conexión con el agente GELLO especificado en los argumentos
        gello_port = args.gello_port
        # Si no se especifica un puerto, se buscan los puertos USB disponibles y se selecciona el primero encontrado para la conexión con el agente GELLO
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            print(f"Found {len(usb_ports)} ports")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"using port {gello_port}")
            # Lanzar un error si no se encuentra ningún puerto USB para la conexión con el agente GELLO
            else:
                raise ValueError(
                    "No gello port found, please specify one or plug in gello"
                )
        # Configuración del agente GELLO con el puerto encontrado y la posición inicial de las articulaciones (si se proporciona)
        agent_cfg = {
            "_target_": "gello.agents.gello_agent.GelloAgent",
            "port": gello_port,
            "start_joints": args.start_joints,
        }
        # En caso de que no se proporcionen las posiciones iniciales de las articulaciones, se define una posición de reinicio predeterminada para el robot.
        if args.start_joints is None:
            reset_joints = np.deg2rad([-180, 0, -180, 0, -270])
        else:
        # Se convierten las posiciones iniciales de las articulaciones en un array de numpy.
            reset_joints = np.array(args.start_joints)
        # Se obtiene la posición actual de las articulaciones del robot a través del entorno.
        curr_joints = env.get_obs()["joint_positions"]
        # Si las posiciones de reinicio y las posiciones actuales de las articulaciones tienen la misma forma:
        if reset_joints.shape == curr_joints.shape:
           # Se obtiene la máxima diferencia entre las posiciones actuales y las posiciones de reinicio 
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            # Se calcula el número de pasos necesarios para mover el robot desde su posición actual hasta la posición de reinicio, limitando el movimiento a un máximo de 0.01 por paso y un máximo de 100 pasos.
            steps = min(int(max_delta / 0.01), 100)
            # Se itera a través de los pasos calculados, moviendo el robot gradualmente desde su posición actual hasta la posición de reinicio.
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)
    # En caso de que se elija un agente dummy o ninguno, se configura un agente dummy para motivos de testeo.
    elif args.agent == "dummy" or args.agent == "none":
        agent_cfg = {
            "_target_": "gello.agents.agent.DummyAgent",
            "num_dofs": robot_client.num_dofs(),
        }
    # TODO: se puede agregar un agente de política para aprendizaje por imitación.
    elif args.agent == "policy":
        raise NotImplementedError("add your imitation policy here if there is one")
    # Error en caso de que se proporcione un nombre de agente no válido.
    else:
        raise ValueError("Invalid agent name")
    # Se instancia el agente utilizando la configuración definida.
    agent = instantiate_from_dict(agent_cfg)
    print("Going to start position")
    # Se obtiene la posición inicial de las articulaciones del robot a través del agente.
    start_pos = agent.act(env.get_obs())
    # Se obtiene la posición actual de las articulaciones del robot a través del entorno.
    obs = env.get_obs()
    # Se extraen las posiciones de las articulaciones relevantes (las primeras 5) del entorno para compararlas con la posición inicial proporcionada por el agente.
    # TODO: Cambiar esto si el número de articulaciones relevantes cambia, o si se quiere comparar todas las articulaciones.
    joints = obs["joint_positions"][:5]
    # Se calcula la diferencia absoluta entre las posiciones iniciales proporcionadas por el agente y las posiciones actuales de las articulaciones del robot.
    abs_deltas = np.abs(start_pos - joints)
    # Se identifica el índice de la articulación con la mayor diferencia entre la posición inicial proporcionada por el agente y la posición actual del robot.
    id_max_joint_delta = np.argmax(abs_deltas)
    # Se define un umbral máximo para la diferencia entre las posiciones iniciales proporcionadas por el agente y las posiciones actuales del robot.
    # TODO: cambiar
    max_joint_delta = 1000 #0.8
    # En caso de que se supere ese umbral no realizara el movimiento y se imprimira el mensaje:
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        # Se obtiene el índice de todas las articulaciones que superan el umbral
        id_mask = abs_deltas > max_joint_delta
        print()
        # Se imprime un mensaje indicando que la posición inicial del agente es demasiado grande, junto con los detalles
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return
    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    # Se asegura que la dimensión de la posición inicial proporcionada por el agente coincide con la dimensión de las articulaciones relevantes del robot.
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
    # Se define un umbral máximo para el cambio permitido en las posiciones de las articulaciones en cada paso.
    max_delta = 0.05
    # Se itera a través de 25 pasos para mover el robot gradualmente desde su posición actual hasta la posición inicial proporcionada por el agente.
    for _ in range(25):
        # Obtiene las posiciones de las articulaciones del robot.
        obs = env.get_obs()
        # Obtiene la acción del agente basada en la observación actual del entorno, que representa las posiciones de las articulaciones deseadas.
        command_joints = agent.act(obs)
        # Se extraen las posiciones actuales de las articulaciones relevantes del entorno para compararlas con las posiciones deseadas proporcionadas por el agente.
        # TODO: cambiar
        current_joints = obs["joint_positions"][:5]
        # Se calcula la diferencia entre las posiciones deseadas proporcionadas por el agente y las posiciones actuales de las articulaciones del robot.
        delta = command_joints - current_joints
        # Se calcula la máxima diferencia entre las posiciones deseadas y las posiciones actuales de las articulaciones.
        max_joint_delta = np.abs(delta).max()
        # En caso de que la máxima diferencia supere el umbral definido, se escala la diferencia para que el cambio en las posiciones de las articulaciones no supere el umbral máximo permitido.
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        # Realiza un paso en el entorno utilizando las posiciones actuales de las articulaciones del robot más la diferencia escalada.
        env.step(current_joints + delta)
    # Obtiene las posiciones de las articulaciones del robot.
    obs = env.get_obs()
    # Extrae los joints 
    joints = obs["joint_positions"]
    # Obtiene la acción del agente
    action = agent.act(obs)
    # 0.5
    # TODO: Cambiar
    # En caso de que la diferencia entre la acción del agente y las posiciones actuales de las articulaciones del robot supere un umbral definido:
    if (action - joints > 1110.5).any():
        print("Action is too big")

        # Imprime un mensaje indicando que la acción del agente es demasiado grande, junto con los detalles de las articulaciones que superan el umbral.
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()
    # Si la acción del agente es válida, se inicia el control teleoperado.
    from gello.utils.control_utils import SaveInterface, run_control_loop
    # En caso de que se active el guardado de datos para el entrenamiento, se inicializa la interfaz de guardado con el directorio de datos especificado en los argumentos, el nombre del agente y la opción de expandir el directorio del usuario.
    save_interface = None
    if args.use_save_interface:
        save_interface = SaveInterface(
            data_dir=args.data_dir, agent_name=args.agent, expand_user=True
        )

    run_control_loop(env, agent, save_interface, use_colors=True)
    

if __name__ == "__main__":
    main(tyro.cli(Args))

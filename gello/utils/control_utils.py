# Importación de librerías necesarias
import datetime
import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
# Importación de módulos específicos de gello
from gello.agents.agent import Agent
from gello.env import RobotEnv
# Maxima diferencia entre joint del agente y robot
DEFAULT_MAX_JOINT_DELTA = 1.0
"""
control_utils.py 
Script de utilidades compartidas para los bucles de control de robots.
"""
"""
Funcion que mueve el robot a la posición inicial de forma gradual.
Args:
    env: RobotEnv, el entorno del robot
    agent: Agent, el agente que proporciona la posición objetivo
    max_delta: float, la máxima diferencia de joint permitida por paso
    steps: int, el número de pasos para llegar a la posición inicial
Returns:
    bool: True si se logró mover a la posición inicial, False si la posición objetivo es demasiado lejana
"""
def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    # Obtener la posición objetivo del agente
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]
    # Calcular la diferencia absoluta entre la posición objetivo y la posición actual
    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)
    # Si la diferencia máxima es mayor que el umbral, imprimir información y retornar False
    max_joint_delta = DEFAULT_MAX_JOINT_DELTA
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
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
        return False
    # Si la diferencia es aceptable, realizar movimientos graduales hacia la posición objetivo
    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"
    # Realizar movimientos graduales hacia la posición objetivo
    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    return True

"""
Clase SaveInterface para manejar la interfaz de guardado basada en teclado.
Permite iniciar y detener la grabación de datos utilizando teclas específicas.
"""
class SaveInterface:
    """
    Initializa la interfaz de guardado.
    Args:
        data_dir: Directorio base para guardar los datos
        agent_name: Nombre del agente (usado para subdirectorio)
        expand_user: Si se debe expandir ~ en la ruta de data_dir
    """
    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
    ):
        # Importar la clase KBReset
        from gello.data_utils.keyboard_interface import KBReset
        # Inicializar la interfaz de teclado y configurar el directorio de guardado
        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        self.save_path: Optional[Path] = None

        print("Save interface enabled. Use keyboard controls:")
        print("  S: Start recording")
        print("  Q: Stop recording")
    """
    Función para actualizar la interfaz de guardado y manejar el proceso de guardado.
    Args:
        obs: Observaciones actuales del entorno
        action: Acción actual tomada por el agente
    Returns:
        Optional[str]: "quit" si el usuario desea salir, None en caso contrario
    """
    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        # Importar la función save_frame para guardar los datos
        from gello.data_utils.format_obs import save_frame
        # Obtener el estado actual de la interfaz de teclado y manejar las acciones correspondientes
        dt = datetime.datetime.now()
        state = self.kb_interface.update()
        # Si el estado es "start", crear un nuevo directorio para guardar los datos con un timestamp
        if state == "start":
            dt_time = datetime.datetime.now()
            self.save_path = (
                self.data_dir / dt_time.strftime("%m%d_%H%M%S")
            )
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {self.save_path}")
        # Si el estado es "save" y hay una ruta de guardado válida, guardar el frame actual utilizando la función save_frame
        elif state == "save":
            if self.save_path is not None:
                save_frame(self.save_path, dt, obs, action)
        # Si el estado es "normal", limpiar la ruta de guardado
        elif state == "normal":
            self.save_path = None
        # Si el estado es "quit", imprimir un mensaje de salida y retornar "quit" para indicar que se desea salir
        elif state == "quit":
            print("\nExiting.")
            return "quit"
        else:
            raise ValueError(f"Invalid state {state}")

        return None

"""
Ejecuta el bucle de control principal para el entorno del robot y el agente dado.
Args:
    env: RobotEnv, el entorno del robot
    agent: Agent, el agente que proporciona las acciones
    save_interface: Optional[SaveInterface], interfaz opcional para manejar el guardado de datos
    print_timing: bool, si se debe imprimir información de tiempo en la consola
    use_colors: bool, si se deben usar colores en la salida de la consola
"""
def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    # Revisa si se pueden usar colores para mejorar la salida de la consola
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)
    # Iniciar el bucle de control principal
    start_time = time.time()
    obs = env.get_obs()
    # Bucle principal de control
    while True:
        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "

            if colors_available:
                print(
                    colored(message, color="white", attrs=["bold"]), end="", flush=True
                )
            else:
                print(message, end="", flush=True)

        action = agent.act(obs)

        # Maneja el proceso de guardado utilizando la interfaz de guardado si está disponible
        if save_interface is not None:
            result = save_interface.update(obs, action)
            if result == "quit":
                break
        # Ejecutar el paso del entorno con la acción actual y obtener las nuevas observaciones
        obs = env.step(action)

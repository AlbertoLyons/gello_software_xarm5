# Importación de librerías
from dataclasses import dataclass
from pathlib import Path
import tyro
# Importación de la clase ZMQServerRobot desde el módulo robot_node
from gello.zmq_core.robot_node import ZMQServerRobot
"""
launch_nodes.py
Script para lanzar el servidor del robot, que puede ser un robot simulado o un robot físico, dependiendo de los argumentos proporcionados. 
El servidor se comunica a través de ZMQ que permite controlar el robot desde otros procesos.
Parámetros asignados desde consola:
- robot: tipo de robot a lanzar (xarm, xarm_no_arm, sim_xarm, sim_xarm_no_arm)
- robot_port: puerto en el que el servidor del robot escuchará las conexiones
- hostname: dirección IP en la que el servidor del robot estará disponible
- robot_ip: dirección IP del robot físico
"""
@dataclass
class Args:
    robot: str = "xarm"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_ip: str = "192.168.123.52" # IP del robot xArm5 físico en la red IoT_v3
"""
Función que lanza el servidor del robot dependiendo del tipo de robot especificado en los argumentos.
"""
def launch_robot_server(args: Args):
    port = args.robot_port
    # Lanza un servidor del robot xArm5 simulado utilizando MuJoCo
    if args.robot == "sim_xarm":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm5" / "xarm5.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        # Inicia el servidor del robot simulado
        server.serve()
    # Lanza un servidor del robot xArm5 simulado sin brazo utilizando MuJoCo
    elif args.robot == "sim_xarm_no_arm":
        from gello.robots.sim_robot import MujocoRobotServer

        MENAGERIE_ROOT: Path = (
            Path(__file__).parent.parent / "mujoco_menagerie"
        )
        xml = MENAGERIE_ROOT / "ufactory_xarm5" / "xarm5_noarm.xml"
        gripper_xml = None
        server = MujocoRobotServer(
            xml_path=xml, gripper_xml_path=gripper_xml, port=port, host=args.hostname
        )
        # Inicia el servidor del robot simulado
        server.serve()
    # Lanza un servidor del robot xArm5 físico
    else:
        # Lanza el servidor del robot con agarre
        if args.robot == "xarm":
            from gello.robots.xarm_robot import XArmRobot
            robot = XArmRobot(ip=args.robot_ip)
        # Lanza el servidor del robot sin agarre
        elif args.robot == "xarm_no_arm":
            from gello.robots.xarm_robot_no_arm import XArmRobot_NoArm
            robot = XArmRobot_NoArm(ip=args.robot_ip)
        else:
            raise NotImplementedError(
                f"Robot {args.robot} not implemented, choose one of: xarm, xarm_no_arm, sim_xarm, sim_xarm_no_arm"
            )
        # Inicia el servidor del robot físico con ZMQ
        server = ZMQServerRobot(robot, port=port, host=args.hostname)
        print(f"Starting robot server on port {port}")
        server.serve()

def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))

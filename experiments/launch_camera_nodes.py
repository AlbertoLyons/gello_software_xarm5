# Importación de las líbrerias necesarias
from dataclasses import dataclass
from multiprocessing import Process
import multiprocessing
import tyro
# Importaión de las clases necesarias para la cámara y el servidor
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.zmq_core.camera_node import ZMQServerCamera

"""
launch_camera_nodes.py
Script que lanza un servidor para cada cámara. 
Cada servidor se ejecuta en un proceso separado y escucha en un puerto diferente.
Parámetros asignados desde consola:
- hostname: dirección IP en la que los servidores de las cámaras estarán disponibles (En este caso se asigna a todos)
"""
@dataclass
class Args:
    # hostname: str = "127.0.0.1"
    hostname: str = "0.0.0.0"

"""
Función que lanza un servidor para cada cámara. Cada servidor se ejecuta en un proceso separado y escucha en un puerto diferente.
"""
def launch_server(port: int, camera_id: int, args: Args):
    camera = RealSenseCamera(camera_id)
    server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()

"""
Función principal que obtiene los IDs de las cámaras disponibles, lanza un servidor para cada cámara en un proceso separado y espera a que terminen.
"""
def main(args):
    # Obtiene los IDs de las cámaras disponibles.
    ids = get_device_ids()
    # Primer puerto para el primer servidor de cámara, se incrementa para cada cámara adicional.
    camera_port = 5000
    camera_servers = []
    # Recorre los IDs de las cámaras y lanza un servidor para cada una, asignándole un puerto diferente.
    for camera_id in ids:
        # Lanza un proceso para cada cámara.
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, args))
        )
        camera_port += 1
    # Inicia todos los procesos de los servidores de las cámaras.
    for server in camera_servers:
        server.start()

# Lanza la función principal con los argumentos obtenidos desde la consola utilizando tyro y multiprocesamiento para ejecutar cada servidor de cámara en un proceso separado.
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main(tyro.cli(Args))

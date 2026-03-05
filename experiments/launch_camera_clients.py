# Importación de bibliotecas necesarias
from dataclasses import dataclass
from typing import Tuple
import tyro
import cv2
# Importa la clase ZMQClientCamera desde el módulo camera_node
from gello.zmq_core.camera_node import ZMQClientCamera
"""
launch_camera_clients.py
Este script lanza clientes de cámara que se conectan a servidores de cámara a través de ZMQ 
lanzados desde launch_camera_nodes.py.
Parametros asignados a través de la línea de comandos:
- ports: Tupla de puertos a los que se conectarán los clientes
- hostname: Dirección IP del host al que se conectarán los clientes
"""
@dataclass
class Args:
    ports: Tuple[int, ...] = (5000,) 
    hostname: str = "127.0.0.1"
    # hostname: str = 192.168.53.152" 
"""
Función principal que inicia los clientes de cámara de Intel Realsense, se conecta a los servidores de cámara y muestra las las cámaras en separados
"""
def main(args):
    # Inicializa los clientes de cámara y las ventanas de visualización
    cameras = []
    images_display_names = []
    # Recorre los puertos especificados en los argumentos
    for port in args.ports:
        # Crea un cliente de cámara para cada puerto y lo agrega a la lista de cámaras
        cameras.append(ZMQClientCamera(port=port, host=args.hostname))
        images_display_names.append(f"Camera_Port_{port}")
        # Crea una ventana de visualización para cada cámara
        cv2.namedWindow(f"Camera_Port_{port}", cv2.WINDOW_NORMAL)
    print("Client started.")
    # Bucle principal para leer las imágenes de las cámaras y mostrarlas en cada ventana la imagen a color, y la imagen de profundidad
    while True:
        # Recorre cada cámara y su correspondiente nombre de visualización
        for display_name, camera in zip(images_display_names, cameras):
            # Lee la imagen y la profundidad de la cámara
            result = camera.read()
            # Si no se obtiene ningún resultado, continúa con la siguiente iteración
            if result is None:
                continue
            # Desempaqueta la imagen y la profundidad del resultado
            image, depth = result
            # Convierte la imagen de RGB a BGR para su visualización con OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Aplica un mapa de colores a la imagen de profundidad para su visualización
            import numpy as np
            depth_fix = np.squeeze(depth)
            depth_colormap = cv2.applyColorMap(depth_fix, cv2.COLORMAP_JET)
            # Combina la imagen de la cámara y el mapa de colores de profundidad en una sola imagen para mostrar
            canvas = cv2.hconcat([image_bgr, depth_colormap])
            # Muestra la imagen combinada en la ventana correspondiente
            cv2.imshow(display_name, canvas)
        # Si se presiona la tecla 'q', sale del bucle y cierra las ventanas
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Cierra todas las ventanas de visualización
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(tyro.cli(Args))
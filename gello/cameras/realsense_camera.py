# Importación de librerías necesarias
import os
import time
from typing import List, Optional, Tuple
import numpy as np
# Importa la clase CameraDriver desde el módulo camera
from gello.cameras.camera import CameraDriver
"""
realsense_camera.py
Script que implemente el driver para las cámaras intel Realsense.
"""
"""
Función para obtener los IDs de los dispositivos Realsense conectados al sistema.
Devuelve una lista de strings con los IDs de los dispositivos.
"""
def get_device_ids() -> List[str]:
    import pyrealsense2 as rs
    # Resetea los dispositivos
    ctx = rs.context()
    # Consulta los dispositivos conectados
    devices = ctx.query_devices()
    device_ids = []
    # Escanea cada dispositivo, resetea el hardware (descomentar el reset si no se utiliza WSL) y obtiene su ID de serie
    for dev in devices:
        #dev.hardware_reset()
        device_ids.append(dev.get_info(rs.camera_info.serial_number))
    time.sleep(2)
    # Devuelve la lista de IDs de los dispositivos
    return device_ids

"""
Clase RealSenseCamera
Clase que implementa el driver para las cámaras Intel Realsense. Hereda de CameraDriver.
Permite inicializar la cámara, leer frames de imagen y profundidad.
"""
class RealSenseCamera(CameraDriver):
    # Representa a la clase obteniendo el ID de la cámara
    def __repr__(self) -> str:
        return f"RealSenseCamera(device_id={self._device_id})"
    # Incializa la clase con el ID del dispositivo y una opción para voltear la imagen (flip)
    def __init__(self, device_id: Optional[str] = None, flip: bool = False):
        import pyrealsense2 as rs
        import time
        
        self._device_id = device_id
        self._flip = flip
        
        self._pipeline = rs.pipeline()
        config = rs.config()
        self._colorizer = rs.colorizer() 

        if device_id is None:
            ctx = rs.context()
            devices = ctx.query_devices()
            for dev in devices:
                dev.hardware_reset()
            time.sleep(2)
        else:
            config.enable_device(device_id)

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self._pipeline.start(config)
    """
    Lee un frame desde la cámara
    Args:
        img_size: El tamaño de la imagen a devolver. Si es None, se devuelve el tamaño original.
        farthest: La distancia más lejana para mapear a 255.
    Returns:
        np.ndarray: La imagen de color, con forma (H, W, 3)
        np.ndarray: La imagen de profundidad, con forma (H, W, 1)
    """
    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,  # Más lejano: float = 0.12
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Importa la librería OpenCV para procesamiento de imágenes
        import cv2
        # Espera a que se reciban los frames de la cámara
        frames = self._pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() # Frame original (datos)
        
        # Crea una versión coloreada para visualización
        depth_colorized = self._colorizer.colorize(depth_frame)
        depth_image = np.asanyarray(depth_colorized.get_data())
        
        color_image = np.asanyarray(color_frame.get_data())
        # depth_image = cv2.convertScaleAbs(depth_image, alpha=0.03)
        # Si no se especifica un tamaño de imagen, se devuelve la imagen original
        if img_size is None:
            image = color_image[:, :, ::-1]
            depth = depth_image
        # De lo contrario, se redimensiona la imagen al tamaño especificado.
        else:
            image = cv2.resize(color_image, img_size)[:, :, ::-1]
            depth = cv2.resize(depth_image, img_size)
        # Centra la cámara rotando la imagen 180 grados en caso de que se haya especificado la opción de flip
        if self._flip:
            image = cv2.rotate(image, cv2.ROTATE_180)
            depth = cv2.rotate(depth, cv2.ROTATE_180)[:, :, None]
        else:
            depth = depth[:, :, None]
        # Devuelve la imagen de color y la imagen de profundidad
        return image, depth

"""
Función de depuración para leer frames de la cámara y mostrar imágenes de color y profundidad
"""
def _debug_read(camera, save_datastream=False):
    import cv2
    # Crea ventanas para mostrar las imágenes de color y profundidad
    cv2.namedWindow("image")
    cv2.namedWindow("depth")
    counter = 0
    # Si no existen las carpetas para guardar las imágenes, se crean
    if not os.path.exists("images"):
        os.makedirs("images")
    # Si se ha especificado la opción de guardar el datastream, se crea la carpeta correspondiente si no existe
    if save_datastream and not os.path.exists("stream"):
        os.makedirs("stream")
    # Bucle para leer los frames de la cámara
    while True:
        time.sleep(0.1)
        # Lee un frame de la cámara
        image, depth = camera.read()
        # Convierte la imagen de profundidad a una imagen de 3 canales para mostrarla en color
        depth = np.concatenate([depth, depth, depth], axis=-1)
        key = cv2.waitKey(1)
        # Muestra las imágenes de color y profundidad en las ventanas correspondientes
        cv2.imshow("image", image[:, :, ::-1])
        cv2.imshow("depth", depth)
        # Si se presiona la tecla 's', se guardan las imágenes de color y profundidad en la carpeta "images"
        if key == ord("s"):
            cv2.imwrite(f"images/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"images/depth_{counter}.png", depth)
        # Si se ha especificado la opción de guardar el datastream, se guardan las imágenes de color y profundidad en la carpeta "stream" con un contador para diferenciarlas
        if save_datastream:
            cv2.imwrite(f"stream/image_{counter}.png", image[:, :, ::-1])
            cv2.imwrite(f"stream/depth_{counter}.png", depth)
        counter += 1
       # Si se presiona la tecla 'ESC', se sale del bucle y se cierran las ventanas
        if key == 27:
            break
# Función principal para probar la clase RealSenseCamera y la función de depuración _debug_read
if __name__ == "__main__":
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices")
    print(device_ids)
    rs = RealSenseCamera(flip=True, device_id=device_ids[0])
    im, depth = rs.read()
    _debug_read(rs, save_datastream=True)

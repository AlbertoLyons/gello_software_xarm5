# Importación de librerías necesarias
from pathlib import Path
from typing import Optional, Protocol, Tuple
import numpy as np
"""
camera.py
Script que define la interfaz para los drivers de cámara y proporciona implementaciones de ejemplo.
"""
"""
Clase CameraDriver
Es un protocolo que define la interfaz para los drivers de cámara. Se usa para abstraer la cámara del resto del código, permitiendo que diferentes implementaciones de cámara puedan ser utilizadas sin cambiar el código que las utiliza.
"""
class CameraDriver(Protocol):
    """
    Función que lee un frame de la cámara.
    Args:
        img_size: El tamaño de la imagen a devolver. Si es None, se devuelve el tamaño original.
        farthest: La distancia más lejana a mapear a 255.
    Returns:
        np.ndarray: La imagen en color.
        np.ndarray: La imagen de profundidad.
    """
    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...
"""
Clase DummyCamera
Es una implementación de ejemplo de CameraDriver que genera imágenes aleatorias
"""
class DummyCamera(CameraDriver):
    """
    Función que lee un frame de la cámara. Genera imágenes aleatorias para simular una cámara real.
    Args:
        img_size: El tamaño de la imagen a devolver. Si es None, se devuelve el tamaño original.
        farthest: La distancia más lejana a mapear a 255.
    Returns:
        np.ndarray: La imagen en color.
        np.ndarray: La imagen de profundidad.
    """
    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Si no se especifica la imagen, se devuelve una imagen de tamaño 480x640 con valores aleatorios
        if img_size is None:
            return (
                np.random.randint(255, size=(480, 640, 3), dtype=np.uint8),
                np.random.randint(255, size=(480, 640, 1), dtype=np.uint16),
            )
        # En caso contrario, se devuelve una imagen del tamaño especificado con valores aleatorios
        else:
            return (
                np.random.randint(
                    255, size=(img_size[0], img_size[1], 3), dtype=np.uint8
                ),
                np.random.randint(
                    255, size=(img_size[0], img_size[1], 1), dtype=np.uint16
                ),
            )

"""
Clase SavedCamera
Permite cargar imágenes de color y profundidad desde archivos en lugar de una cámara real.
"""
class SavedCamera(CameraDriver):
    # Inicializa la cámara con la ruta a las imágenes de color y profundidad
    def __init__(self, path: str = "example"):
        self.path = str(Path(__file__).parent / path)
        from PIL import Image

        self._color_img = Image.open(f"{self.path}/image.png")
        self._depth_img = Image.open(f"{self.path}/depth.png")
    # Función que lee un frame de la cámara.
    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Si se especifica un tamaño de imagen, se redimensionan.
        if img_size is not None:
            color_img = self._color_img.resize(img_size)
            depth_img = self._depth_img.resize(img_size)
        # En caso contrario, se mantiene
        else:
            color_img = self._color_img
            depth_img = self._depth_img
        # Regresa las imágenes como arrays de numpy. La imagen de profundidad se devuelve en una sola escala.
        return np.array(color_img), np.array(depth_img)[:, :, 0:1]

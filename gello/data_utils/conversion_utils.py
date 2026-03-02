# Importación de bibliotecas necesarias
from typing import Dict
import cv2
import numpy as np
import torch
import transforms3d._gohlketransforms as ttf
"""
conversion_utils.py
Script que define funciones de utilidad para el procesamiento de tensores, manipulación de imágenes y gestión de transformaciones espaciales (Pose).
"""
# Función que convierte una entrada (lista, array de numpy o tensor) a un tensor de PyTorch.
def to_torch(array, device="cpu"):
    if isinstance(array, torch.Tensor):
        return array.to(device)
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device)
    else:
        return torch.tensor(array).to(device)

# Función que convierte un tensor de PyTorch a un array de numpy.
def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    return array

# Función que realiza un recorte central cuadrado a un par de frames de RGB y profundidad.
def center_crop(rgb_frame, depth_frame):
    H, W = rgb_frame.shape[-2:]
    sq_size = min(H, W)

    # Recorte del cuadrado central
    if H > W:
        rgb_frame = rgb_frame[..., int((H - sq_size) / 2) : int((H + sq_size) / 2), :sq_size]
        depth_frame = depth_frame[..., int((H - sq_size) / 2) : int((H + sq_size) / 2), :sq_size]
    elif W < H:
        rgb_frame = rgb_frame[..., :sq_size, int((W - sq_size) / 2) : int((W + sq_size) / 2)]
        depth_frame = depth_frame[..., :sq_size, int((W - sq_size) / 2) : int((W + sq_size) / 2)]

    return rgb_frame, depth_frame

# Función que redimensiona las imágenes RGB y de profundidad a un tamaño cuadrado específico (por defecto 224x224).
def resize(rgb, depth, size=224):
    rgb = rgb.transpose([1, 2, 0])
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = rgb.transpose([2, 0, 1])

    depth = cv2.resize(depth[0], (size, size), interpolation=cv2.INTER_LINEAR)
    depth = depth.reshape([1, size, size])
    return rgb, depth

# Función que limpia el mapa de profundidad eliminando valores NaN e infinitos, y limitando los valores entre un rango mínimo y máximo.
def filter_depth(depth, max_depth=2.0, min_depth=0.0):
    depth[np.isnan(depth)] = 0.0
    depth[np.isinf(depth)] = 0.0
    depth = np.clip(depth, min_depth, max_depth)
    return depth

"""
Función principal de preprocesamiento de observaciones. 
Toma un diccionario con datos crudos del robot (RGB, profundidad, estados) y los normaliza, recorta y empaqueta.
"""
def preproc_obs(
    demo: Dict[str, np.ndarray], joint_only: bool = True
) -> Dict[str, np.ndarray]:
    # Obtención y normalización de canales de imagen (C, H, W)
    rgb_wrist = demo.get(f"wrist_rgb").transpose([2, 0, 1]) * 1.0  # type: ignore
    depth_wrist = demo.get(f"wrist_depth").transpose([2, 0, 1]) * 1.0  # type: ignore
    rgb_base = demo.get("base_rgb").transpose([2, 0, 1]) * 1.0  # type: ignore
    depth_base = demo.get("base_depth").transpose([2, 0, 1]) * 1.0  # type: ignore

    # Aplicación de recorte central, redimensionamiento y filtrado de profundidad
    rgb_wrist, depth_wrist = resize(*center_crop(rgb_wrist, depth_wrist))
    rgb_base, depth_base = resize(*center_crop(rgb_base, depth_base))

    depth_wrist = filter_depth(depth_wrist)
    depth_base = filter_depth(depth_base)

    rgb = np.stack([rgb_wrist, rgb_base], axis=0)
    depth = np.stack([depth_wrist, depth_base], axis=0)

    # Definición de matrices intrínsecas y de cámara (dummies por defecto)
    dummy_cam = np.eye(4)
    K = np.eye(3)

    # Extracción de estados del robot
    qpos, qvel, ee_pos_quat, gripper_pos = (
        demo.get("joint_positions"),  # type: ignore
        demo.get("joint_velocities"),  # type: ignore
        demo.get("ee_pos_quat"),  # type: ignore
        demo.get("gripper_position"),  # type: ignore
    )

    if joint_only:
        state: np.ndarray = qpos
    else:
        state: np.ndarray = np.concatenate([qpos, qvel, ee_pos_quat, gripper_pos[None]])

    return {
        "rgb": rgb,
        "depth": depth,
        "camera_poses": dummy_cam,
        "K_matrices": K,
        "state": state,
    }

"""
Clase Pose que representa una posición y orientación en el espacio 3D.
Utiliza cuaterniones internamente para las rotaciones y proporciona métodos para convertir entre diferentes 
representaciones (Euler, Ángulo-Eje, Matrices) y realizar álgebra de transformaciones.
"""
class Pose(object):
    # Inicia la pose con coordenadas (x, y, z) y cuaterniones (qw, qx, qy, qz).
    def __init__(self, x, y, z, qw, qx, qy, qz):
        self.p = np.array([x, y, z])

        # Internamente se usa la convención [x, y, z, w] para compatibilidad con ttf.
        self.q = np.array([qx, qy, qz, qw])

        # Asegura que la parte escalar del cuaternión sea siempre positiva.
        if self.q[3] < 0:
            self.q *= -1

        self.q = self.q / np.linalg.norm(self.q)

    # Operación de multiplicación para encadenar transformaciones espaciales.
    def __mul__(self, other):
        assert isinstance(other, Pose)
        p = self.p + ttf.quaternion_matrix(self.q)[:3, :3].dot(other.p)
        q = ttf.quaternion_multiply(self.q, other.q)
        return Pose(p[0], p[1], p[2], q[3], q[0], q[1], q[2])

    # Multiplicación por la derecha.
    def __rmul__(self, other):
        assert isinstance(other, Pose)
        return other * self

    # Representación en cadena de texto de la posición y orientación.
    def __str__(self):
        return "p: {}, q: {}".format(self.p, self.q)

    # Calcula la pose inversa (traslación y rotación opuesta).
    def inv(self):
        R = ttf.quaternion_matrix(self.q)[:3, :3]
        p = -R.T.dot(self.p)
        q = ttf.quaternion_inverse(self.q)
        return Pose(p[0], p[1], p[2], q[3], q[0], q[1], q[2])

    # Devuelve la pose en formato [x, y, z, qw, qx, qy, qz].
    def to_quaternion(self):
        q_reverted = np.array([self.q[3], self.q[0], self.q[1], self.q[2]])
        return np.concatenate([self.p, q_reverted])

    # Convierte la rotación a representación de ángulo-eje.
    def to_axis_angle(self):
        angle = 2 * np.arccos(self.q[3])
        angle = angle / np.pi
        if angle > 1:
            angle -= 2

        axis = self.q[:3] / np.linalg.norm(self.q[:3])

        if axis[0] < 0:
            axis *= -1
            angle *= -1

        return np.concatenate([self.p, axis, [angle]])

    # Convierte la rotación a ángulos de Euler normalizados.
    def to_euler(self):
        q = np.array(ttf.euler_from_quaternion(self.q))
        if q[0] > np.pi:
            q[0] -= 2 * np.pi
        if q[1] > np.pi:
            q[1] -= 2 * np.pi
        if q[2] > np.pi:
            q[2] -= 2 * np.pi

        q = q / np.pi
        return np.concatenate([self.p, q, [0.0]])

    # Convierte la pose a una matriz de transformación homogénea de 4x4.
    def to_44_matrix(self):
        out = np.eye(4)
        out[:3, :3] = ttf.quaternion_matrix(self.q)[:3, :3]
        out[:3, 3] = self.p
        return out

    # Método estático que crea una Pose a partir de representación de ángulo-eje.
    @staticmethod
    def from_axis_angle(x, y, z, ax, ay, az, phi):
        phi = phi * np.pi
        p = np.array([x, y, z])
        qw = np.cos(phi / 2.0)
        qx = ax * np.sin(phi / 2.0)
        qy = ay * np.sin(phi / 2.0)
        qz = az * np.sin(phi / 2.0)
        return Pose(p[0], p[1], p[2], qw, qx, qy, qz)

    # Método estático que crea una Pose a partir de ángulos de Euler.
    @staticmethod
    def from_euler(x, y, z, roll, pitch, yaw, _):
        p = np.array([x, y, z])
        roll, pitch, yaw = roll * np.pi, pitch * np.pi, yaw * np.pi
        q = ttf.quaternion_from_euler(roll, pitch, yaw)
        return Pose(p[0], p[1], p[2], q[3], q[0], q[1], q[2])

    # Método estático que crea una Pose a partir de coordenadas y un cuaternión explícito.
    @staticmethod
    def from_quaternion(x, y, z, qw, qx, qy, qz):
        p = np.array([x, y, z])
        return Pose(p[0], p[1], p[2], qw, qx, qy, qz)

"""
Función que calcula la acción inversa (transformación relativa) entre dos poses.
"""
def compute_inverse_action(p, p_new, ee_control=False):
    assert isinstance(p, Pose) and isinstance(p_new, Pose)
    if ee_control:
        dpose = p.inv() * p_new
    else:
        dpose = p_new * p.inv()
    return dpose

"""
Función que calcula la nueva pose resultante de aplicar un delta de movimiento (dpose) a una pose inicial.
"""
def compute_forward_action(p, dpose, ee_control=False):
    assert isinstance(p, Pose) and isinstance(dpose, Pose)
    dpose = Pose.from_quaternion(*dpose.to_quaternion())

    if ee_control:
        p_new = p * dpose
    else:
        p_new = dpose * p

    return p_new
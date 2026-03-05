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
    # Aseguramos que rgb sea (C, H, W)
    if rgb_frame.ndim == 2:
        rgb_frame = rgb_frame[None, ...]
    
    C, H, W = rgb_frame.shape
    sq_size = min(H, W)
    
    start_h = (H - sq_size) // 2
    start_w = (W - sq_size) // 2

    rgb_cropped = rgb_frame[:, start_h : start_h + sq_size, start_w : start_w + sq_size]
    
    # Manejo de profundidad: si es (H, W) o (C, H, W)
    if depth_frame.ndim == 3:
        depth_cropped = depth_frame[:, start_h : start_h + sq_size, start_w : start_w + sq_size]
    else:
        depth_cropped = depth_frame[start_h : start_h + sq_size, start_w : start_w + sq_size]
        # Forzamos a que siempre salga con un canal al menos (1, H, W)
        depth_cropped = depth_cropped[None, ...]

    return rgb_cropped, depth_cropped

# Función que redimensiona las imágenes RGB y de profundidad a un tamaño cuadrado específico (por defecto 224x224).
def resize(rgb, depth, size=224):
    target_size = (int(size), int(size))
    
    if rgb.shape[0] < rgb.shape[1]:
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
    else:
        rgb_hwc = rgb
    
    rgb_res = cv2.resize(rgb_hwc, target_size, interpolation=cv2.INTER_LINEAR)
    rgb_final = np.transpose(rgb_res, (2, 0, 1))

    # Asegurar que sea H, W, C para el resize
    if depth.ndim == 2:
        depth_hwc = depth[:, :, None]
    elif depth.shape[0] < depth.shape[1]:
        depth_hwc = np.transpose(depth, (1, 2, 0))
    else:
        depth_hwc = depth

    #Si tiene 3 canales por el colorizador, colapsar a 1
    if depth_hwc.shape[2] == 3:
        depth_hwc = depth_hwc[:, :, 0:1]

    depth_res = cv2.resize(depth_hwc, target_size, interpolation=cv2.INTER_LINEAR)

    if depth_res.ndim == 2:
        depth_final = depth_res[None, ...]
    else:
        depth_final = np.transpose(depth_res.reshape(size, size, 1), (2, 0, 1))

    return rgb_final.astype(np.float32), depth_final.astype(np.float32)

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
    
    wrist_rgb_raw = demo.get("wrist_rgb")
    wrist_depth_raw = demo.get("wrist_depth")
    
    if wrist_rgb_raw is not None:
        rgb_wrist = wrist_rgb_raw.transpose([2, 0, 1]) * 1.0
        depth_wrist = wrist_depth_raw.transpose([2, 0, 1]) * 1.0
        
        rgb_wrist, depth_wrist = resize(*center_crop(rgb_wrist, depth_wrist))
        depth_wrist = filter_depth(depth_wrist)
    else:
        rgb_wrist = np.zeros((3, 224, 224), dtype=np.float32)
        depth_wrist = np.zeros((1, 224, 224), dtype=np.float32)

    rgb_base = np.zeros_like(rgb_wrist)
    depth_base = np.zeros_like(depth_wrist)

    rgb = np.stack([rgb_wrist, rgb_base], axis=0)
    depth = np.stack([depth_wrist, depth_base], axis=0)

    qpos = demo.get("joint_positions")
    qvel = demo.get("joint_velocities")
    ee_pos_quat = demo.get("ee_pos_quat")
    gripper_pos = demo.get("gripper_position")

    # En casos de que el gripper no exista
    if gripper_pos is None:
        gripper_pos = np.array([0.0], dtype=np.float32)
    elif isinstance(gripper_pos, (float, int)):
        gripper_pos = np.array([gripper_pos], dtype=np.float32)
    elif hasattr(gripper_pos, 'ndim') and gripper_pos.ndim == 0:
        gripper_pos = gripper_pos[None]

    if qpos is None:
        raise ValueError("Error: 'joint_positions' not found in the data.")

    if joint_only:
        state = qpos
    else:
        parts = [qpos]
        if qvel is not None: parts.append(qvel)
        if ee_pos_quat is not None: parts.append(ee_pos_quat)
        parts.append(gripper_pos)
        state = np.concatenate(parts)

    return {
        "rgb": rgb.astype(np.uint8), 
        "depth": depth.astype(np.float32),
        "camera_poses": np.eye(4).astype(np.float32),
        "K_matrices": np.eye(3).astype(np.float32), 
        "state": state.astype(np.float32),
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
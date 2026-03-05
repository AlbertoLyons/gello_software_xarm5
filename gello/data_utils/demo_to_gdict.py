# Importación de bibliotecas necesarias
import glob
import os
import pickle
import shutil
from dataclasses import dataclass
import sys
from typing import Tuple, List
import numpy as np
import tyro
from natsort import natsorted
from tqdm import tqdm
import h5py
from gello.data_utils.plot_utils import plot_in_grid
np.set_printoptions(precision=3, suppress=True)

import mediapy as mp
sys.path.append(os.getcwd())
from gdict.data import DictArray, GDict
from gello.data_utils.conversion_utils import preproc_obs

"""
demo_to_gdict.py
Script para procesar demostraciones recolectadas en archivos .pkl y convertirlas a formato HDF5.
El script calcula factores de escala y bias para normalizar las acciones, preprocesa las observaciones
de las cámaras y genera visualizaciones automáticas de los datos procesados.
"""
# Función auxiliar para guardar un diccionario anidado en formato HDF5.
def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        _recursively_save_dict_contents_to_group(h5file, '/', dic)

def _recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        key = str(key)
        if isinstance(item, dict):
            _recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            data = np.array(item)
            try:
                h5file.create_dataset(path + key, data=data, compression="gzip")
            except TypeError:
                h5file.create_dataset(path + key, data=data)
# Función que calcula los valores mínimos y máximos de las acciones (control) en un directorio de demostración.
def get_act_min_max(source_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    pkls = natsorted(
        glob.glob(os.path.join(source_dir, "**/*.pkl"), recursive=True), reverse=True
    )
    if len(pkls) <= 30:
        print(f"Skipping {source_dir} because it has less than 30 frames.")
        raise RuntimeError("Too few frames")
    pkls = pkls[:-5]

    scale_min = None
    scale_max = None
    for pkl in pkls:
        try:
            with open(pkl, "rb") as f:
                demo = pickle.load(f)
        except Exception as e:
            print(f"Skipping {pkl} because it is corrupted.")
            print(f"Error: {e}")
            raise Exception("Corrupted pkl")

        requested_control = demo.pop("control")
        curr_scale_factor = requested_control
        if scale_min is None:
            assert scale_max is None
            scale_min = curr_scale_factor
            scale_max = curr_scale_factor
        else:
            assert scale_max is not None
            scale_min = np.minimum(scale_min, curr_scale_factor)
            scale_max = np.maximum(scale_max, curr_scale_factor)
    assert scale_min is not None
    assert scale_max is not None
    return scale_min, scale_max

"""
Función que procesa una única demostración.
1. Convierte los archivos .pkl en un objeto DictArray y luego a HDF5.
2. Normaliza las acciones utilizando los factores de escala y bias proporcionados.
3. Genera y guarda videos (RGB/Depth) y gráficas de estado/acción para inspección visual.
"""
def convert_single_demo(
    source_dir,
    i,
    traj_output_dir,
    rgb_output_dir,
    depth_output_dir,
    state_output_dir,
    action_output_dir,
    scale_factor,
    bias_factor,
):
    pkls = natsorted(
        glob.glob(os.path.join(source_dir, "**/*.pkl"), recursive=True), reverse=True
    )
    demo_stack = []

    if len(pkls) <= 30:
        return 0

    pkls = pkls[:-5]

    for pkl in pkls:
        curr_ts = {}
        try:
            with open(pkl, "rb") as f:
                demo = pickle.load(f)
        except:
            print(f"Skipping {pkl} because it is corrupted.")
            return 0

        obs = preproc_obs(demo)
        action = demo.pop("control")
        action = (action - bias_factor) / scale_factor  # Normalización entre -1 y 1

        curr_ts["obs"] = obs
        curr_ts["actions"] = action
        curr_ts["dones"] = np.zeros(1)  
        curr_ts["episode_dones"] = np.zeros(1)

        curr_ts_wrapped = dict()
        curr_ts_wrapped[f"traj_{i}"] = curr_ts
        demo_stack = [curr_ts_wrapped] + demo_stack

    keys = demo_stack[0][f"traj_{i}"].keys()
    demo_dict = {
        "actions": np.array([d[f"traj_{i}"]["actions"] for d in demo_stack], dtype=np.float32),
        "dones": np.array([d[f"traj_{i}"]["dones"] for d in demo_stack], dtype=bool),
        "episode_dones": np.array([d[f"traj_{i}"]["episode_dones"] for d in demo_stack], dtype=bool),
    }

    # 2. Handle the 'obs' dictionary separately to ensure sub-keys are stacked correctly
    obs_keys = demo_stack[0][f"traj_{i}"]["obs"].keys()
    demo_dict["obs"] = {}
    
    for k in obs_keys:
        # We force convert to a clean numpy array. 
        # If this line fails, it means your .pkl files have inconsistent shapes for that key.
        stacked_obs = np.stack([d[f"traj_{i}"]["obs"][k] for d in demo_stack])
        demo_dict["obs"][k] = stacked_obs

    # --- END OF REPLACEMENT ---

    output_file = os.path.join(traj_output_dir, f"traj_{i}.h5")
    save_dict_to_hdf5(demo_dict, output_file)
    print(f"Saved trajectory {i} to {output_file}")
    # Base Camera (Index 1)
    all_rgbs_base = demo_dict["obs"]["rgb"][:, 1].transpose([0, 2, 3, 1])
    all_rgbs_base = all_rgbs_base.astype(np.uint8)
    _, H, W, _ = all_rgbs_base.shape
    
    all_depths_base = demo_dict["obs"]["depth"][:, 1].reshape([-1, H, W])
    all_depths_base = all_depths_base / 5.0  

    mp.write_video(
        os.path.join(rgb_output_dir, f"traj_{i}_rgb_base.mp4"), all_rgbs_base, fps=30
    )
    mp.write_video(
        os.path.join(depth_output_dir, f"traj_{i}_depth_base.mp4"),
        all_depths_base,
        fps=30,
    )

    # Wrist Camera (Index 0)
    all_rgbs_wrist = demo_dict["obs"]["rgb"][:, 0].transpose([0, 2, 3, 1])
    all_rgbs_wrist = all_rgbs_wrist.astype(np.uint8)
    
    all_depths_wrist = demo_dict["obs"]["depth"][:, 0].reshape([-1, H, W])
    all_depths_wrist = all_depths_wrist / 2.0  

    mp.write_video(
        os.path.join(rgb_output_dir, f"traj_{i}_rgb_wrist.mp4"), all_rgbs_wrist, fps=30
    )
    mp.write_video(
        os.path.join(depth_output_dir, f"traj_{i}_depth_wrist.mp4"),
        all_depths_wrist,
        fps=30,
    )

    # Prep for Return and Grids
    all_actions = demo_dict["actions"]
    all_states = demo_dict["obs"]["state"]

    curr_actions = all_actions.reshape([1, *all_actions.shape])
    curr_states = all_states.reshape([-1, *all_states.shape])

    plot_in_grid(
        curr_actions, os.path.join(action_output_dir, f"traj_{i}_actions.png")
    )
    plot_in_grid(
        curr_states, os.path.join(state_output_dir, f"traj_{i}_states.png")
    )

    # To maintain consistency with the tile logic in your original snippet
    all_depths_tiled = np.tile(all_depths_wrist[..., None], [1, 1, 1, 3])

    return all_rgbs_wrist, all_depths_tiled, all_actions, all_states

@dataclass
class Args:
    source_dir: str
    vis: bool = True

# Función principal que gestiona la conversión masiva de demostraciones.
def main(args):
    subdirs = natsorted(glob.glob(os.path.join(args.source_dir, "*/"), recursive=True))

    output_dir = args.source_dir
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    output_dir = os.path.join(output_dir, "_conv")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_dir = os.path.join(output_dir, "multiview")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    else:
        print(f"Output directory {output_dir} already exists, and will be deleted")
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    if not os.path.isdir(val_dir):
        os.mkdir(val_dir)

    val_size = int(min(0.1 * len(subdirs), 10))
    val_indices = np.random.choice(len(subdirs), size=val_size, replace=False)
    val_indices = set(val_indices)

    # Cálculo de factores de escala para normalización global
    print("Computing scale factors")
    pbar = tqdm(range(len(subdirs)))
    min_scale_factor = None
    max_scale_factor = None
    for i in pbar:
        try:
            curr_min, curr_max = get_act_min_max(subdirs[i])
            if min_scale_factor is None:
                assert max_scale_factor is None
                min_scale_factor = curr_min
                max_scale_factor = curr_max
            else:
                assert max_scale_factor is not None
                min_scale_factor = np.minimum(min_scale_factor, curr_min)
                max_scale_factor = np.maximum(max_scale_factor, curr_max)
            pbar.set_description(f"t: {i}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Skipping {subdirs[i]}")
            continue
            
    bias_factor = (min_scale_factor + max_scale_factor) / 2.0
    scale_factor = (max_scale_factor - min_scale_factor) / 2.0
    scale_factor[scale_factor == 0] = 1.0
    
    print("*" * 80)
    print(f"scale factors: {scale_factor}")
    print(f"bias factor: {bias_factor}")
    scale_factor_str = ", ".join([f"{x}" for x in scale_factor])
    print(f"scale_factor = np.array([{scale_factor_str}])")
    bias_factor_str = ", ".join([f"{x}" for x in bias_factor])
    print(f"bias_factor = np.array([{bias_factor_str}])")
    print("*" * 80)

    tot = 0
    all_rgbs = []
    all_depths = []
    all_actions = []
    all_states = []

    # Configuración de carpetas de visualización
    vis_dir = os.path.join(output_dir, "vis")
    state_output_dir = os.path.join(vis_dir, "state")
    action_output_dir = os.path.join(vis_dir, "action")
    rgb_output_dir = os.path.join(vis_dir, "rgb")
    depth_output_dir = os.path.join(vis_dir, "depth")

    if not os.path.isdir(vis_dir):
        os.mkdir(vis_dir)
    if not os.path.isdir(state_output_dir):
        os.mkdir(state_output_dir)
    if not os.path.isdir(action_output_dir):
        os.mkdir(action_output_dir)
    if not os.path.isdir(rgb_output_dir):
        os.mkdir(rgb_output_dir)
    if not os.path.isdir(depth_output_dir):
        os.mkdir(depth_output_dir)

    # Fase de conversión y guardado de trayectorias
    pbar = tqdm(range(len(subdirs)))
    for i in pbar:
        out_dir = val_dir if i in val_indices else train_dir
        out_dir = os.path.join(out_dir, "none")

        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        ret = convert_single_demo(
            subdirs[i],
            i,
            out_dir,
            rgb_output_dir,
            depth_output_dir,
            state_output_dir,
            action_output_dir,
            scale_factor=scale_factor,
            bias_factor=bias_factor,
        )

        if ret != 0:
            all_rgbs.append(ret[0])
            all_depths.append(ret[1])
            all_actions.append(ret[2])
            all_states.append(ret[3])
            tot += 1

        pbar.set_description(f"t: {i}")

    print(
        f"Finished converting all demos to {output_dir}! (num demos: {tot} / {len(subdirs)})"
    )

    # Generación de visualizaciones agregadas si se solicita
    if args.vis:
        if len(all_rgbs) > 0:
            print(f"Visualizing all demos...")

            plot_in_grid(
                all_actions, os.path.join(action_output_dir, "_all_actions.png")
            )
            plot_in_grid(all_states, os.path.join(state_output_dir, "_all_states.png"))
            make_grid_video_from_numpy(
                all_rgbs, 10, os.path.join(rgb_output_dir, "_all_rgb.mp4"), fps=30
            )
            make_grid_video_from_numpy(
                all_depths, 10, os.path.join(depth_output_dir, "_all_depth.mp4"), fps=30
            )

    exit(0)

# Función que toma una lista de videos, los organiza en una cuadrícula y exporta el archivo mp4 resultante.
def make_grid_video_from_numpy(
    video_list: List[np.ndarray], 
    cols: int, 
    output_path: str, 
    fps: int = 30
):
    if not video_list:
        return

    min_frames = min(v.shape[0] for v in video_list)
    num_videos = len(video_list)
    rows = int(np.ceil(num_videos / cols))
    _, h, w, c = video_list[0].shape
    
    grid_frames = []
    for t in range(min_frames):
        frames_t = [v[t] for v in video_list]
        while len(frames_t) < rows * cols:
            frames_t.append(np.zeros((h, w, c), dtype=np.uint8))
        
        grid_rows = []
        for r in range(rows):
            row_concat = np.concatenate(frames_t[r * cols : (r + 1) * cols], axis=1)
            grid_rows.append(row_concat)
        
        full_grid = np.concatenate(grid_rows, axis=0)
        grid_frames.append(full_grid)
    
    mp.write_video(output_path, np.array(grid_frames), fps=fps)
    print(f"Video saved in: {output_path}")

if __name__ == "__main__":
    main(tyro.cli(Args))
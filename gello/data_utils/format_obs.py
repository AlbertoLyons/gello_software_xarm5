# Importación de liberías necesarias
import datetime
import pickle
from pathlib import Path
from typing import Dict
import numpy as np

"""
format_obs.py
Script para formatear las observaciones y guardarlas en archivos pickle.
"""
"""
Función para guardar un frame de observación y acción en un archivo pickle.
- folder: Path donde se guardará el archivo.
- timestamp: Marca de tiempo para nombrar el archivo.
- obs: Diccionario de observaciones.
- action: Acción correspondiente a la observación.
"""
def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
) -> None:
    obs["control"] = action  # Agregar la acción al diccionario de observaciones

    # Crear archivo pickle para guardar las observaciones y acciones
    folder.mkdir(exist_ok=True, parents=True)
    recorded_file = folder / (timestamp.isoformat() + ".pkl")

    with open(recorded_file, "wb") as f:
        pickle.dump(obs, f)

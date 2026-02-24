# Importación de librerías necesarias
import importlib
"""
launch_utils.py
Este script proporciona la infraestructura para orquestar el sistema GELLO.
Gestiona la carga de configuraciones, la instanciación de robots,
la configuración de nodos de comunicación ZMQ y el bucle de control principal.
"""
# Instancia objetos dinámicamente desde el diccionario de configuración usando '_target_'.
def instantiate_from_dict(cfg):
    
    if isinstance(cfg, dict) and "_target_" in cfg:
        module_path, class_name = cfg["_target_"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
        # Recursión para procesar posibles sub-objetos en los argumentos
        return cls(**{k: instantiate_from_dict(v) for k, v in kwargs.items()})
    elif isinstance(cfg, dict):
        return {k: instantiate_from_dict(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [instantiate_from_dict(v) for v in cfg]
    else:
        return cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True,)
    args = parser.parse_args()


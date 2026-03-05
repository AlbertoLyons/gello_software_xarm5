#!/bin/bash

# Configuración de rutas
BASE_DIR="/mnt/c/Users/Beto/Desktop/Informatica/PRACTICA/gello_software_xarm5"
VENV_PATH=".venv/bin/activate"

# Función para limpiar pantalla y mostrar encabezado
show_header() {
    clear
    echo "========================================="
    echo "       $1"
    echo "========================================="
}

# --- MENU CÁMARA ---
MENU_CAMARA() {
    show_header "Selección Uso de Cámara"
    echo "  1. Iniciar con cámara funcionando"
    echo "  2. Iniciar sin cámara"
    echo "========================================="
    read -p "Selecciona una opcion (1-2): " opcion

    case $opcion in
        1)
            echo "--- NODO CÁMARA ---"
            gnome-terminal -- bash -c "cd $BASE_DIR; source $VENV_PATH; python3 experiments/launch_camera_nodes.py; exec bash" &
            MENU_XARM
            ;;
        2)
            MENU_XARM
            ;;
        *)
            MENU_CAMARA
            ;;
    esac
}

# --- MENU XARM ---
MENU_XARM() {
    show_header "Selección Sistema xArm5"
    echo "  1. Iniciar entorno simulado"
    echo "  2. Iniciar brazo real"
    echo "========================================="
    read -p "Selecciona una opcion (1-2): " opcion

    case $opcion in
        1)
            echo "--- CONEXIÓN CON XARM5 SIMULADO ---"
            gnome-terminal -- bash -c "cd $BASE_DIR; source $VENV_PATH; python3 experiments/launch_nodes.py --robot sim_xarm_no_arm; exec bash" &
            MENU_GELLO
            ;;
        2)
            echo "--- CONEXIÓN CON XARM5 ---"
            gnome-terminal -- bash -c "cd $BASE_DIR; source $VENV_PATH; python3 experiments/launch_nodes.py --robot xarm_no_arm; exec bash" &
            MENU_GELLO
            ;;
        *)
            MENU_XARM
            ;;
    esac
}

# --- MENU GELLO ---
MENU_GELLO() {
    show_header "Selección Modalidad GELLO"
    echo "  1. Iniciar con guardado de datos"
    echo "     (tecla s para empezar, tecla q para terminar)"
    echo "  2. Iniciar sin guardado de datos"
    echo "========================================="
    read -p "Selecciona una opcion (1-2): " opcion

    echo "Esperando a que se inicie el brazo..."
    sleep 20

    case $opcion in
        1)
            echo "Iniciando agente GELLO con guardado de datos"
            cd "$BASE_DIR"
            source "$VENV_PATH"
            python3 experiments/run_env.py --agent=gello --use-save-interface
            exit
            ;;
        2)
            echo "Iniciando agente GELLO"
            cd "$BASE_DIR"
            source "$VENV_PATH"
            python3 experiments/run_env.py --agent=gello
            exit
            ;;
        *)
            MENU_GELLO
            ;;
    esac
}

# Iniciar el script
MENU_CAMARA
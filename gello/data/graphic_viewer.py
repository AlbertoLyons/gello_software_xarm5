# Importación de librerías necesarias
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

# La carpeta donde se encuentran los archivos .pkl deben de estar en el mismo del script
folder = "./0218_152448" 
files = sorted([f for f in os.listdir(folder) if f.endswith('.pkl')])

print(f"Processing {len(files)} files...")

times = []
joint_data = []
gripper_data = []

# Recorre los files, extrayendo los datos de joints y gripper
for i, name in enumerate(files):
    path = os.path.join(folder, name)
    if os.path.getsize(path) == 0: continue

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            # Verifica que el archivo contenga la llave 'joint_positions' antes de acceder a ella
            if isinstance(data, dict) and 'joint_positions' in data:
                joint_data.append(data['joint_positions'])
                times.append(i)
                gripper_data.append(data.get('gripper_position', 0))
    except Exception:
        continue

# Empieza a crear las gráficas solo si se extrajeron datos correctamente
if not joint_data:
    print("Cannot extract data.")
else:
    array_data = np.array(joint_data)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Gráfica de Joints
    for j in range(array_data.shape[1]):
        ax1.plot(times, array_data[:, j], label=f'J{j+1}')
    ax1.set_title('Joint positions (xArm5)')
    ax1.set_ylabel('Radians')
    ax1.legend(loc='right')
    ax1.grid(True)

    # Gráfica de Gripper
    ax2.plot(times, gripper_data, color='black', label='Gripper')
    ax2.set_title('Gripper position (xArm5)')
    ax2.set_xlabel('Sample (File)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
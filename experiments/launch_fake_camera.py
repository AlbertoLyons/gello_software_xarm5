from dataclasses import dataclass
from typing import Tuple
import numpy as np
import tyro
import zmq
import time
import cv2

@dataclass
class Args:
    ports: Tuple[int, ...] = (5000, 5001)

def main(args):
    context = zmq.Context()
    publishers = []
    
    for port in args.ports:
        socket = context.socket(zmq.PUB)
        socket.bind(f"tcp://*:{port}")
        publishers.append(socket)
        print(f"✓ Fake camera server started on port {port}")
    
    time.sleep(1)
    print("Sending fake camera data... Press Ctrl+C to stop")
    
    frame_count = 0
    try:
        while True:
            for i, pub in enumerate(publishers):
                # Crear imagen fake
                color = [(100, 50, 200), (50, 200, 100)][i]
                fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
                fake_image[:, :] = color
                
                # Añadir texto
                cv2.putText(fake_image, f"Camera {i} Frame {frame_count}", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Crear depth fake
                fake_depth = np.ones((480, 640), dtype=np.uint8) * 128
                
                # Enviar en formato BGR (como espera OpenCV)
                pub.send_pyobj({
                    'image': fake_image,
                    'depth': fake_depth
                })
            
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS
    except KeyboardInterrupt:
        print("\nStopping fake camera servers...")

if __name__ == "__main__":
    main(tyro.cli(Args))
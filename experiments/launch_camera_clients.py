from dataclasses import dataclass
from typing import Tuple
import numpy as np
import tyro
import cv2
from gello.zmq_core.camera_node import ZMQClientCamera

@dataclass
class Args:
    ports: Tuple[int, ...] = (5000,) 
    hostname: str = "127.0.0.1"
    # hostname: str = 192.168.53.152" 

def main(args):
    cameras = []
    images_display_names = []
    
    for port in args.ports:
        cameras.append(ZMQClientCamera(port=port, host=args.hostname))
        images_display_names.append(f"Camera_Port_{port}")
        cv2.namedWindow(f"Camera_Port_{port}", cv2.WINDOW_NORMAL)

    print("Client started.")

    while True:
        for display_name, camera in zip(images_display_names, cameras):
            result = camera.read()
            
            if result is None:
                continue
            image, depth = result
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            canvas = cv2.hconcat([image_bgr, depth_colormap])
            cv2.imshow(display_name, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(tyro.cli(Args))
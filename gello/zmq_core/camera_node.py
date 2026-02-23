import pickle
import threading
from typing import Optional, Tuple

import numpy as np
import zmq

from gello.cameras.camera import CameraDriver

DEFAULT_CAMERA_PORT = 5000


class ZMQClientCamera(CameraDriver):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_CAMERA_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        # pack the image_size and send it to the server
        send_message = pickle.dumps(img_size)
        self._socket.send(send_message)
        state_dict = pickle.loads(self._socket.recv())
        return state_dict


class ZMQServerCamera:
    def __init__(
        self,
        camera,
        port: int = 5000,
        host: str = "0.0.0.0",
    ):
        self._camera = camera
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        
        addr = f"tcp://{host}:{port}"
        print(f"Binding Camera Server to {addr}")
        self._socket.bind(addr)
        
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._lock = threading.Lock()
        
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        """Este hilo mantiene el buffer de la RealSense siempre vacío."""
        while not self._stop_event.is_set():
            try:
                frame = self._camera.read() 
                with self._lock:
                    self._latest_frame = frame
            except Exception as e:
                print(f"Error in frame: {e}")
                time.sleep(0.1)

    def serve(self) -> None:
        print("Camera server ready")
        while not self._stop_event.is_set():
            try:
                if self._socket.poll(1000): 
                    message = self._socket.recv()
                    img_size = pickle.loads(message)
                    
                    with self._lock:
                        if self._latest_frame is not None:
                            self._socket.send(pickle.dumps(self._latest_frame))
                        else:
                            self._socket.send(pickle.dumps(None))
            except Exception as e:
                print(f"Error: {e}")

    def stop(self) -> None:
        self._stop_event.set()
        self._capture_thread.join()
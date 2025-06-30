import numpy as np
import struct
import os


def load_images(file_name):
    dir = "./data"
    file_path = os.path.join(dir, file_name)
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
        return (images / 255.0).tolist()  # Normalize


def load_labels(file_name):
    dir = "./data"
    file_path = os.path.join(dir, file_name)
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels.tolist()

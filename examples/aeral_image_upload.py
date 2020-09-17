from PIL import Image
import numpy as np

from hub.areal.storage_tensor import StorageTensor


def main():
    image = Image.open("./data/very_big_image.jpg")
    arr = np.array(image)
    tensor = StorageTensor(
        "s3://snark-hub/bigtensor", shape=arr.shape, dtype=arr.dtype, memcache=100
    )
    print(f"Shape: {arr.shape}, Dtype: {arr.dtype}")
    tensor[:] = arr[:]


if __name__ == "__main__":
    main()
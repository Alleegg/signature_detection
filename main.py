from PIL import Image 
import numpy as np
import transformations


if __name__ == "__main__":

    imge = "signature_detection/file_image/Поглазова conv 3.png"

    yolo = transformations.YOLOv8_detect(image_path = imge)
    trans = transformations.mask(yolo)
    filtered = transformations.filter_clusters(trans)

    for i in filtered:
        Image.fromarray((i * 255).astype(np.uint8)).show()


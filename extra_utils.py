import cv2
from pathlib import Path
from datetime import datetime
import numpy as np
from yolov5.utils.general import is_docker, is_colab, LOGGER


def path_with_date(path, mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    date = datetime.utcnow()
    date_str = f"{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}"
    path = path.joinpath(date_str)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def calculate_distance(pixel_height, focal_length, average_height, cls):
    pixel_height = float(pixel_height)
    cls = float(cls)
    min_height = average_height - 0.2
    max_height = average_height + 0.2
    if cls == 0:  # human
        average_height = average_height
        min_height = min_height
        max_height = max_height
    elif cls == 2:  # car
        average_height = 1.5
        min_height = 1.4
        max_height = 1.9
    elif cls == 5:  # bus
        average_height = 3.0
        min_height = 2.5
        max_height = 4.0
    elif cls == 7:  # truck
        average_height = 3.0
        min_height = 2.0
        max_height = 4.0
    else:
        average_height = 0
    if average_height != 0:
        d = focal_length * average_height / pixel_height
        d_min = focal_length * min_height / pixel_height
        d_max = focal_length * max_height / pixel_height
    else:
        d = 0
        d_min = 0
        d_max = 0
    return d, d_min, d_max


# THIS PART IS NOT NEEDED RIGHT NOW!
# THIS IS CHECKING CAMERA WITH JETSON UTILS!
def check_imshow_jetson():
    import jetson.utils
    # Check if environment supports image displays
    try:
        # assert not is_docker(), 'cv2.imshow() is disabled in Docker environments'
        assert not is_colab(), 'cv2.imshow() is disabled in Google Colab environments'
        output = jetson.utils.videoOutput("display://0")
        input = jetson.utils.videoSource("csi://0")
        image = input.Capture(format='rgb8')
        output.Render(image)
        # cv2.imshow('test', np.zeros((1, 1, 3)))
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)
        return True
    except Exception as e:
        LOGGER.warning(f'WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays\n{e}')
        return False
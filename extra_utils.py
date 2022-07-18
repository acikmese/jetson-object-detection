import cv2
import os
import shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from yolov5.utils.general import is_docker, is_colab, LOGGER


def path_with_date(path, date_time, mkdir=False):
    path = Path(path)  # os-agnostic
    date = date_time
    date_str = f"{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}"
    path = path.joinpath(date_str)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path, date_str


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
        d = round(focal_length * average_height / pixel_height, 1)
        d_min = round(focal_length * min_height / pixel_height, 1)
        d_max = round(focal_length * max_height / pixel_height, 1)
    else:
        d = 0
        d_min = 0
        d_max = 0
    return d, d_min, d_max


def zip_with_datetime(results_dir, tmp_dir, output_dir, time_obj):
    try:
        tmp_dir_w_date, time_str = path_with_date(tmp_dir, time_obj)
        tmp_dir_w_date.mkdir(parents=True, exist_ok=True)
        all_files = os.listdir(results_dir)
        for f in all_files:
            shutil.move(results_dir / f, tmp_dir_w_date / f)
        shutil.make_archive(tmp_dir_w_date, 'zip', tmp_dir_w_date)
        shutil.move(str(tmp_dir_w_date) + ".zip", str(output_dir / time_str) + ".zip")
        all_tmp_files = os.listdir(tmp_dir)
        for tmp_f in all_tmp_files:
            shutil.rmtree(tmp_dir / tmp_f, ignore_errors=True)
        return True, f"Successfully zipped and transferred. {time_str}"
    except Exception as e:
        return False, e


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
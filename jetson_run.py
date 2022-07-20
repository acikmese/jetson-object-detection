import cv2
import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import torch
import torch.backends.cudnn as cudnn

# Set framework path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
YOLO_ROOT = os.path.join(ROOT, "yolov5")  # Update ROOT path because of submodule.
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))  # add ROOT to PATH
YOLO_ROOT = Path(os.path.relpath(YOLO_ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, check_img_size, check_imshow,
                                  non_max_suppression, print_args, scale_coords,
                                  xyxy2xywh)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, time_sync
from streamers import LoadCSI, LoadWebcam
from extra_utils import calculate_distance, zip_with_datetime


@torch.no_grad()
def run(weights=YOLO_ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=YOLO_ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        nocsi=False,  # indicates that CSI camera is not used, USB or URL is used
        data=YOLO_ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        project=ROOT / 'output',  # save results to project/name
        name='dets',  # save results to project/name
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference
        focal_length=1000,  # focal length of camera in pixels
        add_distance=True,  # add distance information for some classes
        avg_height=1.75,  # average height of human being (for distance calculation)
        save_img=True,  # Save images in given interval
        img_save_interval=60,  # in seconds
        annotate_img=True,  # Save annotated images or raw images
        zip_files=True,  # Zip files and transfer to given path
        zip_files_interval=300,  # in seconds
        zip_txt_name='zipped_data',  # name of directory to save zipped txt output
        zip_log_name='zipped_logs',  # name of directory to save zipped log data
        zip_img_name='zipped_images',  # name of directory to save zipped images
        ):
    # Set sources
    source = str(source)

    # Set paths
    # Set directories to save output
    txt_dir = Path(project) / name / 'txt'  # txt output dir
    log_dir = Path(project) / name / 'logs'  # log output dir
    img_dir = Path(project) / name / 'images'  # image output dir
    # Set directories to send zipped data
    zip_txt_dir = ROOT / zip_txt_name  # Where to put zipped text files
    zip_log_dir = ROOT / zip_log_name  # Where to put zipped images
    zip_img_dir = ROOT / zip_img_name  # Where to put logs
    # Set temporary directories
    tmp_txt_dir = Path(project) / name / 'tmp_txt_zips'  # temp txt path
    tmp_log_dir = Path(project) / name / 'tmp_log_zips'  # temp logs path
    tmp_img_dir = Path(project) / name / 'tmp_img_zips'  # temp images path

    # Create paths for saving outputs
    if not nosave:
        txt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        if save_img:  # Set image save directory
            img_dir.mkdir(parents=True, exist_ok=True)
        # If it will zip files, it generates temp and final paths for txt, images and log files.
        if zip_files:
            # Create temp and final directories
            tmp_txt_dir.mkdir(parents=True, exist_ok=True)
            tmp_log_dir.mkdir(parents=True, exist_ok=True)
            zip_txt_dir.mkdir(parents=True, exist_ok=True)
            zip_log_dir.mkdir(parents=True, exist_ok=True)
            if save_img:  # Set temp and final zip directories for images
                tmp_img_dir.mkdir(parents=True, exist_ok=True)
                zip_img_dir.mkdir(parents=True, exist_ok=True)
        # Save logs to specified path
        log_file_handler = logging.FileHandler(str(log_dir / "logs") + ".log", mode='a')
        log_file_handler.setLevel(logging.INFO)
        log_formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
        log_file_handler.setFormatter(log_formatter)
        LOGGER.addHandler(log_file_handler)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    view_img = check_imshow() if view_img else False
    cudnn.benchmark = True  # set True to speed up constant image size inference
    if nocsi:  # if we are not using CSI camera. (mostly for development purposes on PC)
        dataset = LoadWebcam(source, img_size=imgsz, stride=stride, auto=pt)  # to use Webcam or URL
    else:
        dataset = LoadCSI(source, img_size=imgsz, stride=stride, auto=pt)  # to use CSI camera
    bs = len(dataset)  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    current_time = datetime.now(timezone.utc)
    utc_prev_time = dict()  # collect time for each camera
    focal = dict()  # collect focal length of each camera
    for i, s in enumerate(dataset.sources):
        source_name = Path(s).stem  # get camera name
        utc_prev_time[source_name] = current_time
        # Standard focal length of camera is described for 1080p resolution,
        # but if it has different resolution, focal length should be updated accordingly.
        if dataset.imgs[i].shape[0] != 1080:
            focal[source_name] = focal_length * (dataset.imgs[i].shape[0] / 1080)
        else:
            focal[source_name] = focal_length
    zip_timer = time_sync()  # timer for zipping files
    first_run = True  # Check if it is first run
    first_run_zip = False  # Check if it needs to zip files in first run

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        # Get datetime of current image.
        utc_time = datetime.now(timezone.utc)
        utc_iso = utc_time.isoformat()

        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)
        t4 = time_sync()
        dt[2] += time_sync() - t3

        # Calculate fps.
        fps = int(1 / (t4 - t1))

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{str(p)}: '

            p = Path(p)  # to Path
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            # Checks if there is a person in image. System won't save image if there is no person in image.
            person_detected = False

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    d = 0  # Set distance as zero for default.
                    if add_distance:
                        # Distance
                        obj_height = xyxy[3] - xyxy[1]  # Calculate length of height of bounding box of detected object
                        d = calculate_distance(obj_height, focal[p.stem], avg_height, cls)

                    if view_img or annotate_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        lbl = f"{names[c]} | {conf:.2f} | {d[0]:.1f}m" if add_distance else f"{names[c]} | {conf:.2f}"
                        label = None if hide_labels else (names[c] if hide_conf else lbl)
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # Save for production txt file.
                    # Converts xyxy points to x_center, y_center and width and height
                    xywh_norm = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  # unnormalized xywh
                    c = int(cls)
                    label = names[c]
                    confidence = round(float(conf), 2)
                    line = [utc_iso, frame, label, c, confidence, *xywh, *xywh_norm, *d]  # label format
                    prod_text_path = str(txt_dir / p.stem) + ".txt"
                    if not os.path.isfile(prod_text_path) or os.stat(prod_text_path).st_size == 0:
                        with open(prod_text_path, 'a') as f:
                            out_txt = f"datetime_utc,frame_id,id_type,id,confidence,x_center,y_center,x_width,y_width,"\
                                      f"x_norm_center,y_norm_center,x_norm_width,y_norm_width," \
                                      f"avg_distance,min_distance,max_distance\n"
                            f.write(out_txt)
                    with open(prod_text_path, 'a') as f:
                        txt = ','.join(map(str, line))
                        f.write(txt + '\n')
                    if c == 0:  # If a person (class:0) is detected, let system know there is person in image to save.
                        person_detected = True

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # If it save images and there is a person,
            # it checks time interval from previous image save.
            # If it is another camera, it saves because time is already passed for previous camera.
            if save_img and person_detected and (utc_time - utc_prev_time[p.stem]).total_seconds() >= img_save_interval:
                it = utc_time
                f_name = f"{p.stem}_{it.year}_{it.month}_{it.day}_{it.hour}_{it.minute}_{it.second}"
                img_path_name = str(img_dir) + '/' + f_name + '.jpg'
                cv2.imwrite(img_path_name, im0, [cv2.IMWRITE_JPEG_QUALITY, 80])
                utc_prev_time[p.stem] = utc_time

        # If it is first run and there is an output, let system zip files.
        if not nosave and first_run:
            if len(os.listdir(txt_dir)) > 0:  # if there is a file
                for file in os.listdir(txt_dir):
                    file_name = str(txt_dir / file)
                    if Path(file_name).stat().st_size >= 1000000:  # if any file has more than 1MB
                        first_run_zip = True
            first_run = False

        # Check if time passed to zip files
        if zip_files and (first_run_zip or (time_sync() - zip_timer >= zip_files_interval)):
            # Zip and move txt output
            txt_success, txt_msg = zip_with_datetime(txt_dir, tmp_txt_dir, zip_txt_dir, utc_time)
            LOGGER.info(f"TXT zip and transfer process: {txt_success}: {txt_msg}")
            # Zip and move log output
            log_success, log_msg = zip_with_datetime(log_dir, tmp_log_dir, zip_log_dir, utc_time)
            LOGGER.info(f"LOG zip and transfer process: {log_success}: {log_msg}")
            # Zip and move image output
            if save_img and len(os.listdir(img_dir)) > 0:  # if there is image output
                img_success, img_msg = zip_with_datetime(img_dir, tmp_img_dir, zip_img_dir, utc_time)
                LOGGER.info(f"IMAGE zip and transfer process: {img_success}: {img_msg}")
            zip_timer = time_sync()
            first_run_zip = False

        # One loop time
        t5 = time_sync()
        fps_loop = int(1 / (t5 - t1))

        # Print time (inference-only)
        LOGGER.info(f"{s}done. ({t3 - t2:.3f}s) ({fps}fps) (loop:{t5 - t4:.3f}s) ({fps_loop}fps)")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT/'prod_model/model.engine', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT/'streams.txt', help='file/dir/URL/glob, 0 for camera')
    parser.add_argument('--data', type=str, default=YOLO_ROOT/'data/coco128.yaml', help='(optional) dataset.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 2, 5, 7], help='filter by class: --classes 0 2')
    parser.add_argument('--name', default='ff', help='save results to project/name')
    parser.add_argument('--focal-length', default=1000, type=int, help='focal length of camera in pixels')
    parser.add_argument('--avg-height', default=1.75, type=float, help='average height of human in meters')
    parser.add_argument('--nocsi', action='store_true', help='indicates that CSI camera is not used, USB or URL is used')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

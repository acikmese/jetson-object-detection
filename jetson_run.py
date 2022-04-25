import cv2
import argparse
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime, timezone
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
YOLO_ROOT = os.path.join(ROOT, "yolov5")  # Update ROOT path because of submodule.
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))  # add ROOT to PATH
YOLO_ROOT = Path(os.path.relpath(YOLO_ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, check_img_size, check_imshow, colorstr,
                                  increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                  xyxy2xywh)
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device, time_sync
from streamers import LoadCSI, LoadWebcam
from extra_utils import path_with_date, calculate_distance

# THESE MAY NOT NEEDED!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "15"


@torch.no_grad()
def run(weights=YOLO_ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=YOLO_ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=YOLO_ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        update=False,  # update all models
        project=ROOT / 'output',  # save results to project/name
        name='dets',  # save results to project/name
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference
        focal_length=1000,  # focal length of camera in pixels
        add_distance=False,  # add distance information for some classes
        avg_height=1.75,  # average height of human being (for distance calculation)
        ):
    source = str(source)
    save_img = True
    zip_files = True
    annotate_img = True
    zip_files_interval = 1 * 10  # in seconds
    img_save_interval = 1  # in seconds
    zip_txt_dir = ROOT / 'zip_txt_final'
    zip_img_dir = ROOT / 'zip_img_final'

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    view_img = check_imshow() if view_img else False
    cudnn.benchmark = True  # set True to speed up constant image size inference
    # dataset = LoadCSI(source, img_size=imgsz, stride=stride, auto=pt)
    dataset = LoadWebcam(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size

    # Focal length calibration.
    focal = None

    utc_prev_time = datetime.now(timezone.utc)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    first_run = True  # if first run, create directories for output
    new_dir_created = False  # if new directory created, it will zip folder and move to output path

    zip_timer = datetime.now()

    if zip_files:
        txt_zip_dir = Path(project) / name / 'tmp_txt_zips'
        txt_zip_dir.mkdir(parents=True, exist_ok=True)
        if save_img:
            img_zip_dir = Path(project) / name / 'tmp_img_zips'
            img_zip_dir.mkdir(parents=True, exist_ok=True)

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        # Get datetime of current image.
        utc_time = datetime.now(timezone.utc)
        utc_iso = utc_time.isoformat()

        # Generate directory
        zip_timer_diff = (datetime.now() - zip_timer).total_seconds()

        if (not nosave) and ((zip_timer_diff >= zip_files_interval) or first_run):
            save_dir, dir_name = path_with_date(Path(project) / name, utc_time)  # output with date
            txt_dir = save_dir.joinpath('txt')
            img_dir = save_dir.joinpath('images')

            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            txt_dir.mkdir(parents=True, exist_ok=True)
            if save_img:
                img_dir.mkdir(parents=True, exist_ok=True)

            if not first_run:
                new_dir_created = True
            else:
                first_run = False
                old_save_dir = deepcopy(save_dir)
                old_txt_dir = deepcopy(txt_dir)
                old_img_dir = deepcopy(img_dir)
                old_dir_name = deepcopy(dir_name)

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
        fps = 1 / (t4 - t1)
        print(f"FPS: {fps}")

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            # Update focal length according to resolution.
            if focal is None:
                img_height = im0.shape[0]
                if img_height != 1080:
                    focal = focal_length * (img_height / 1080)
                else:
                    focal = focal_length

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
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

                    # Distance
                    obj_height = xyxy[3] - xyxy[1]  # Calculate length of height of bounding box of detected object
                    d = calculate_distance(obj_height, focal, avg_height, cls)

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
                    confidence = float(conf)
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

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            if save_img and (utc_time - utc_prev_time).total_seconds() >= img_save_interval:
                it = utc_time
                f_name = f"{p.stem}_{it.year}_{it.month}_{it.day}_{it.hour}_{it.minute}_{it.second}"
                img_path_name = str(img_dir) + '/' + f_name + '.jpg'
                cv2.imwrite(img_path_name, im0, [cv2.IMWRITE_JPEG_QUALITY, 80])
                utc_prev_time = utc_time

        if zip_files and new_dir_created:
            zip_txt_path = txt_zip_dir / old_dir_name
            shutil.move(old_txt_dir, zip_txt_path)
            shutil.make_archive(zip_txt_path, 'zip', zip_txt_path)
            shutil.rmtree(zip_txt_path, ignore_errors=True)
            shutil.move(str(zip_txt_path) + ".zip", str(zip_txt_dir / old_dir_name) + ".zip")
            if save_img:
                zip_img_path = img_zip_dir / old_dir_name
                shutil.move(old_img_dir, zip_img_path)
                shutil.make_archive(zip_img_path, 'zip', zip_img_path)
                shutil.rmtree(zip_img_path, ignore_errors=True)
                shutil.move(str(zip_img_path) + ".zip", str(zip_img_dir / old_dir_name) + ".zip")
            shutil.rmtree(old_save_dir, ignore_errors=True)
            old_dir_name = dir_name
            old_save_dir = save_dir
            old_txt_dir = txt_dir
            old_img_dir = img_dir
            new_dir_created = False
            zip_timer = datetime.now()

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=YOLO_ROOT / 'yolov5n.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'streams.txt', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=YOLO_ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save texts')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 2, 5, 7], help='filter by class: --classes 0 2')
    parser.add_argument('--name', default='ff', help='save results to project/name')
    parser.add_argument('--focal-length', default=1000, type=int, help='focal length of camera in pixels')
    parser.add_argument('--avg-height', default=1.75, type=float, help='average height of human in meters')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
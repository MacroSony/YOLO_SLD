import sys
from pathlib import Path

import torch

from pathlib import Path
import sys

import numpy as np

# Assuming 'yolov9' is a directory within the 'src' directory of your project
ROOT = Path(__file__).resolve().parents[2]  # 'src' directory
YOLO_ROOT = ROOT / 'yolov9'  # YOLOv9 root directory

if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))  # add YOLO_ROOT to PATH

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.general import (LOGGER, Profile, check_img_size,
                           non_max_suppression, scale_boxes)
from yolov9.utils.torch_utils import select_device, smart_inference_mode

from yolov9.utils.augmentations import letterbox


def load_model(weights='yolo.pt', device='cpu', imgsz=(640, 640), half=False, dnn=False):
    device = select_device(device)
    model = DetectMultiBackend(weights, dnn=dnn, device=device, fp16=half)
    model.warmup(imgsz=(1, 3, *imgsz))
    return model


@smart_inference_mode()
def inference(model, imgs, imgsz=(640, 640), 
              conf_thres=0.25, iou_thres=0.45, max_det=1000, 
              classes=None, agnostic_nms=False, 
              augment=False, visualize=False) -> dict:
    stride, names = model.stride, model.names

    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    dt = (Profile(), Profile(), Profile())
    preds = {}
    max_conf = float('-inf')

    for j, im0 in enumerate(imgs):
        
        im = letterbox(im0, imgsz, stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        dets = {'img': im0, 'bboxes': []}
        # Convert Results
        for det in pred:  # per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Append Results
                for *xyxy, conf, cls in reversed(det):
                    if conf > max_conf:
                        max_conf = conf
                        preds["max"] = j
                    dets['bboxes'].append([cls, conf, *xyxy]) # [class, confidence, x1, y1, x2, y2]
        preds[j] = dets
    return preds

from src.utils import pdf_to_png, crop_images, draw_bboxes_xyxy, non_max_suppression, patches_bbox_to_img_bbox,count_with_conf_thresholds
from src.detect.detect import load_model, inference_all_patches
from src.detect.template_match import get_template, template_match
from patchify import unpatchify
import numpy as np
import cv2

imgs = pdf_to_png(directory=".") # List of PIL images

patches = crop_images(images=imgs, crop_size=1200, step=1100) # List of list of patches in ndarray

page_to_check = {0}

print(imgs[0].width, imgs[0].height)
print(patches[0].shape)
    
model = load_model(weights='./models/640_v2_s.pt', device='cpu', imgsz=(640, 640), half=False) # pytorch model

preds = inference_all_patches(patches, page_to_check, model, conf_thres=0.6) # Dictionary of predictions

all_bboxes = patches_bbox_to_img_bbox(preds, 1100)

all_bboxes = non_max_suppression(np.array(all_bboxes), 0.5)

img = draw_bboxes_xyxy(np.array(imgs[0]), all_bboxes, color=(199, 0, 0), thickness=3)
cv2.imwrite("output.png", img)

print(count_with_conf_thresholds(all_bboxes, [0.6, 0.7, 0.8, .85, 0.9]))
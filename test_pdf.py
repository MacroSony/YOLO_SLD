from src.utils import pdf_to_png, crop_images
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

# preds = []
# for i, patch in enumerate(patches):
#     if i in page_to_check:
#         for row in patch:
#             preds.append(inference(model, row, conf_thres=0.6)) # Dictionary of predictions

# template = get_template(preds) # np array

# all_locs = []
# for i, img in enumerate(imgs):
#     locs = template_match(template, np.asarray(img), name=f"{i}", threshold=0.7)
#     all_locs.append(locs)

# print(all_locs)
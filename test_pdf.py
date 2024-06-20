from src.utils import pdf_to_png, crop_images
from src.detect.detect import load_model, inference
from src.detect.template_match import get_template, template_match
import numpy as np

imgs = pdf_to_png(directory=".") # List of PIL images

patches = crop_images(images=imgs, crop_size=1200, step=600) # List of patches

model = load_model(weights='./models/640_v2.pt', device='cpu', imgsz=(640, 640), half=False) # pytorch model

preds = inference(model, patches) # Dictionary of predictions

template = get_template(preds) # np array

all_locs = []
for i, img in enumerate(imgs):
    locs = template_match(template, np.asarray(img), name=f"{i}", threshold=0.7)
    all_locs.append(locs)

print(all_locs)
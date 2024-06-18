from src.utils import pdf_to_png, crop_images
from src.detect.detect import load_model, inference
import numpy as np

imgs = pdf_to_png(directory=".")

patches = crop_images(images=imgs, crop_size=1200, step=600)

model = load_model(weights='./models/640_v2.pt', device='cpu', imgsz=(640, 640), half=False)

preds = inference(model, patches)
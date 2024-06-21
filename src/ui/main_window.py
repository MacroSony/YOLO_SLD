import gradio as gr

from src.utils import pdf_to_png, crop_images, draw_bboxes_xyxy, non_max_suppression, patches_bbox_to_img_bbox,count_with_conf_thresholds
from src.detect.detect import load_model, inference_all_patches
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile

def find_breakers(page_to_check, file_path, model):
    file_path = Path(file_path)
    imgs = pdf_to_png(directory=file_path.parent) # List of PIL images

    patches = crop_images(images=imgs, crop_size=1200, step=1100) # List of list of patches in ndarray
    if page_to_check == "":
        page_set_to_check = set(range(len(imgs)))
    else:
        page_set_to_check = set(map(int, page_to_check.split(",")))

    output_imgs = []
    all_bboxes = []
    preds = inference_all_patches(patches, page_set_to_check, model, conf_thres=0.6) # Dictionary of predictions
    for page in page_set_to_check:
        bboxes = patches_bbox_to_img_bbox(preds[page], 1100)
        bboxes = non_max_suppression(np.array(bboxes), 0.5)
        all_bboxes.extend(bboxes)
        img = draw_bboxes_xyxy(np.array(imgs[page]), bboxes, color=(199, 0, 0), thickness=3, names=model.names)
        output_imgs.append(resize_images_to_long_edge(Image.fromarray(img), 3000))

    return output_imgs, format_output_counts(count_with_conf_thresholds(all_bboxes, [0.6, 0.7, 0.8, .85, 0.9]))

def resize_images_to_long_edge(img, long_edge_size):
    # resized_images = []
    # Calculate the new size maintaining the aspect ratio
    ratio = max(img.size) / long_edge_size
    new_size = (int(img.size[0] / ratio), int(img.size[1] / ratio))

    # Resize the image and append to the list
    resized_img = img.resize(new_size)
    return resized_img

def format_output_counts(counts):
    line1 = f"<h2 style='display: inline'> <span style='color:red'> {counts[0.6]} </span> Breakers Detected </h2> <p style='display: inline'>with Confidence >= 60%</p>"
    line2 = "<p style='color:grey'> "
    for conf, count in counts.items():
        if conf == 0.6:
            continue
        line2 += f"{count} with Confidence >= {conf*100}%, "
    line2 = line2[:-2] + "</p>"
    return line1 + line2

def main():
    model = load_model(weights='./models/640_v2_s.pt', device='cpu', imgsz=(640, 640), half=False) # pytorch model

    def find_breakers_wrapper(page_to_check, file):
        imgs, counts = find_breakers(page_to_check, file, model)
        image_paths = []
        for img in imgs:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                img.save(tmp.name, format='PNG')
                image_paths.append(tmp.name)
        return image_paths, counts
    
    demo = gr.Interface(
        fn=find_breakers_wrapper,
        inputs=[gr.Textbox(label="Pages to Check"), gr.File(file_types=["pdf"], label="Upload SLD")],
        outputs=[gr.Gallery(label="Detected Breakers", interactive=False), "html"],\
        allow_flagging=False
    )

    demo.launch()
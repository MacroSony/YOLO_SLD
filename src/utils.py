import albumentations as A
import cv2
import fitz
import os

import numpy as np
from patchify import patchify
from PIL import Image

def bbox_to_pixel_space(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    # Convert to top-left x, y, bottom-right x, y
    x_min = int(x_center - (width / 2))
    y_min = int(y_center - (height / 2))
    x_max = int(x_center + (width / 2))
    y_max = int(y_center + (height / 2))

    return x_min, y_min, x_max, y_max

def draw_bboxes_yolo(image, bboxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image. Boxes are expected in YOLO format.

    Parameters:
    - image: The input image.
    - bboxes: A list of bounding boxes, each one in YOLO format (x_center, y_center, width, height).
    - color: The color of the boxes. Default is green.
    - thickness: Line thickness. Default is 2.
    """
    img = image.copy()
    img_height, img_width = img.shape[:2]

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox_to_pixel_space(bbox, img_width, img_height)
        img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

    return img

def transform(images, bboxes, transfomration, augmentation_factor):
    transformed_images = []
    for img, bbox in zip(images, bboxes):
        for i in range(augmentation_factor):
            transformed_img = transfomration(image=img, bboxes=bbox)
            transformed_images.append(transformed_img)

    return transformed_images

def write_images(images, bboxes, name, file_path):
    for i, img in enumerate(images):
        cv2.imwrite(f'{file_path}/{name}_{i}.png', img)
        with open(f'{file_path}/{name}_{i}.txt', "w") as f:
            for bbox in bboxes:
                f.write("0 "+" ".join(map(str, bbox)) + "\n")
 
def pdf_to_png(directory, output_directory=None):
    # Iterate over all files in the directory
    imgs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            # Construct input and output paths
            input_path = os.path.join(directory, filename)
            # Open the PDF file
            pdf_document = fitz.open(input_path)
            # Iterate over each page in the PDF
            for page_number in range(len(pdf_document)):
                # Get the page
                page = pdf_document.load_page(page_number)
                # Convert the page to a high-quality PNG image
                pixmap = page.get_pixmap(dpi=300, alpha=False)
                if output_directory:
                    # Determine the output filename
                    output_filename = f"{os.path.splitext(filename)[0]}_{page_number + 1}.png"
                    if output_directory is None:
                        output_path = os.path.join(directory, output_filename)
                    else:
                        output_path = os.path.join(output_directory, output_filename)
                    # Save the PNG image
                    pixmap.save(output_path, "png")
                imgs.append(Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples))
    return imgs

def crop_image(image, crop_size, step=None):
    """
    Crop an image to squares with specified size
    """ 
    img_arr = np.asarray(image) 

    if step is None:
        step = crop_size

    patches = patchify(img_arr, (crop_size, crop_size,3), step=crop_size)
    return patches

def crop_images(images, crop_size, step=None):
    """
    Crop all images in a directory to squares with specified size
    """
    patches = []
    if step is None:
        step = crop_size
    for img in images:
        patches.append(crop_image(img, crop_size, step).reshape(-1, crop_size, crop_size, 3))
    
    return np.array(patches).reshape(-1, crop_size, crop_size, 3)
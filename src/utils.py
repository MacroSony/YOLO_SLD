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

def draw_bboxes_xyxy(image, bboxes, color=(0, 255, 0), thickness=2):
    img = image.copy()
    img_height, img_width = img.shape[:2]

    for bbox in bboxes:
        cls, conf, x_min, y_min, x_max, y_max = bbox
        img = cv2.rectangle(img, (int(x_min.item()), int(y_min.item())), (int(x_max.item()), int(y_max.item())), color, thickness)

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

    patches = patchify(img_arr, (crop_size, crop_size,3), step=step)
    return patches

def crop_images(images, crop_size, step=None):
    """
    Crop all images in a directory to squares with specified size
    """
    all_patches = []
    if step is None:
        step = crop_size
    for img in images:
        all_patches.append(crop_image(img, crop_size, step).squeeze(2))
    return all_patches    

def crop_image_xyxy(image, xyxy):

    """
    Crop an image to a bounding box
    """
    img_arr = np.asarray(image) 
    x_min, y_min, x_max, y_max = xyxy
    return img_arr[y_min:y_max, x_min:x_max]

def template_match(template, img, name, threshold=0.8):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray, template_grey, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    h, w = template_grey.shape

    img_copy = img.copy()
    boxes = []
    for pt in zip(*loc[::-1]):

        boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
    
    boxes = non_max_suppression(np.array(boxes), 0.5)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img_copy,(x1, y1), (x2, y2), (0,0,255), 2)
    
    cv2.imwrite(f'res_{name}.png',img_copy)
    print(boxes)
    return boxes

# Completely GPT, but it works.
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    # If the bounding boxes are integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Initialize the list of picked indexes
    pick = []

    # Grab the coordinates of the bounding boxes
    x1 = boxes[:,2]
    y1 = boxes[:,3]
    x2 = boxes[:,4]
    y2 = boxes[:,5]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # Grab the last index in the indexes list and add the index
        # value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find the largest (x, y) coordinates for the start of the
        # bounding box and the smallest (x, y) coordinates for the end
        # of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have overlap greater
        # than the provided threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # Return only the bounding boxes that were picked
    return boxes[pick]

def patch_bbox_to_img_bbox(patch_bboxes, patch_xy, step):
    img_bboxes = []
    for bbox in patch_bboxes:
        cls, conf, x1, y1, x2, y2 = bbox
        img_bboxes.append([cls, conf, x1 + patch_xy[0] * step, y1 + patch_xy[1] * step, x2 + patch_xy[0] * step, y2 + patch_xy[1] * step])
    return img_bboxes

def patches_bbox_to_img_bbox(patches_bboxes, step):
    img_bboxes = []
    for y in range(len(patches_bboxes)):
        for x in patches_bboxes[y].keys():
            if x != "max":
                img_bboxes.extend(patch_bbox_to_img_bbox(patches_bboxes[y][x]["bboxes"], (x, y), step))
    return img_bboxes

def count_with_conf_thresholds(bboxes, thresholds):
    return {threshold: len([bbox for bbox in bboxes if bbox[1] > threshold]) for threshold in thresholds}
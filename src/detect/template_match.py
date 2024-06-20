import cv2 as cv
import numpy as np

from src.utils import crop_image_xyxy

def get_template(preds):
    bboxes = preds[preds["max"]]["bboxes"]
    best_bbox = bboxes[0]
    for bbox in bboxes[1:]:
        if bbox[1] > best_bbox[1]:
            best_bbox = bbox
    xyxy = [int(tensor.item()) for tensor in best_bbox[2:]]
    template = crop_image_xyxy(preds[preds["max"]]["img"], xyxy)
    cv.imwrite("template.png", template)
    return template


def template_match(template, img, name, threshold=0.8):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    template_grey = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    res = cv.matchTemplate(img_gray, template_grey, cv.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    h, w = template_grey.shape

    img_copy = img.copy()
    boxes = []
    for pt in zip(*loc[::-1]):

        boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
    
    boxes = non_max_suppression(np.array(boxes), 0.5)

    for x1, y1, x2, y2 in boxes:
        cv.rectangle(img_copy,(x1, y1), (x2, y2), (0,0,255), 2)
    
    cv.imwrite(f'res_{name}.png',img_copy)
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
    return boxes[pick].astype("int")

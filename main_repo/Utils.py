
import os
import supervision as sv
import torch


from groundingdino.util.inference import Model

from segment_anything import sam_model_registry, SamPredictor

from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import time
from Config import  GROUNDING_DINO_CHECKPOINT_PATH, GROUNDING_DINO_CONFIG_PATH ,BOX_TRESHOLD, TEXT_TRESHOLD  , SAM_CHECKPOINT_PATH, SAM_ENCODER_VERSION,font_path

from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies

import imutils
import math
from collections import Counter


#torch config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Decorator : calculate execution time
from functools import wraps
from time import time

def execution_time(func):
    @wraps(func)
    def time_it(*args,**kwargs):
        start_time = time()
        try:
            return func(*args,**kwargs)
        finally:
            end = time() - start_time
            print(f"Function '{func.__name__}' execution time: {end}s")
    return time_it




#DINO config
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)



#add all + s to all classes to detect all object of the same class not only one class
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


@execution_time
def detect_annotate_dino(image , classes):
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=classes),
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )


    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    bboxes = detections.xyxy.tolist()

    return detections , annotated_frame ,bboxes


################################################################
#SAM
################################################################

#SAM SETUP
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam.to(device="cpu")

sam_predictor = SamPredictor(sam)


def segment(sam_predictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


@execution_time
def segmentation(detections , sam_predictor , image, classes):
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return(annotated_image)


################################################################
#Crop the image
################################################################

@execution_time
def crop_image(detections , output_directory , image):
    croped_images=[]
    for idx, mask in enumerate(detections.mask):

        segmentation_mask = detections.mask[idx]
        binary_mask = np.where(segmentation_mask > 0.5, 1, 0)

        white_background = np.ones_like(image) * 255
        new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]

        medicament_filename = os.path.join(output_directory, f"medicament{idx}.png")
        cv2.imwrite(medicament_filename, new_image)

        croped_images.append(new_image)

    return croped_images



################################################################
#test
################################################################
def test(images):
    for rot in images:
        rot_reshape = np.sum(rot, axis=2)
        print(rot_reshape.shape)

    #image = images[0]
    #image = image.astype("uint8")
    #gray = cv2.cvtColor(image , cv2.COLOR_BGRA2GRAY)
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #lines = cv2.HoughLines(edges, 3, np.pi / 180, 200)
    #angles = []
    #for line in lines:
        #rho, theta = line[0]
        #angles.append(theta)
    #print(angles)

################################################################
#rotate
################################################################

# @execution_time
# def rotate_image(images,output_directory):

#     rot_images=[]
#     for idx,rot_i in enumerate(images):
#         rot_i_reshape = rot_i.astype(np.uint8)
#         gray = cv2.cvtColor(rot_i_reshape , cv2.COLOR_BGRA2GRAY)

#         edges = cv2.Canny(rot_i_reshape, 50, 150)
#         lines = cv2.HoughLines(edges, 3, np.pi / 180, 200)

#         angles = []
#         for line in lines:
#             rho, theta = line[0]
#             angles.append(theta)

      

#         dominant_angle = np.median(angles)
#         dominant_angle_degrees = dominant_angle * 180 / np.pi

#         height, width = gray.shape
#         center = (width // 2, height // 2)

#         rotation_matrix = cv2.getRotationMatrix2D(center, dominant_angle_degrees, 1.0)
        
#         rotated_image = cv2.warpAffine(rot_i_reshape, rotation_matrix, (width, height))


#         rot_images.append(rotated_image)

#         rot_file = os.path.join(output_directory, f"rotated_image{idx}.png")
#         cv2.imwrite(rot_file, rotated_image)

#     return rot_images

# @execution_time
# def get_text_orientation(images):
#     orientations = []
#     for idx, rot_i in enumerate(images):
#         image = rot_i.astype(np.uint8)

#         gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

#         height, width = gray.shape
#         mask = np.zeros((height, width), np.uint8)

#         contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#         contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

#         cv2.drawContours(mask, contours, 0, (255, 0, 0), 1)
#         edges = cv2.Canny(mask, 50, 150)
#         lines = cv2.HoughLines(edges, 3, np.pi/180, 200)

#         list_angles = []
#         try:
#             for r_theta in lines:
#                 arr = np.array(r_theta[0], dtype=np.float64)
#                 r, theta = arr
#                 list_angles.append(theta)
#             orientations.append(list_angles)
#         except:
#             orientations.append([0, 0, 0])
        
#     return orientations

# @execution_time
# def rotate_image(images, orientations,output_directory):
#     rotated_images = []

#     for idx, (img, list_angles) in enumerate(zip(images, orientations)):
#         RGB_image = img.astype(np.uint8)
#         elements = Counter(list_angles)
#         keys = elements.most_common(1)
#         angle = keys[0][0]
#         angle = angle * 180 / math.pi
#         print(angle)

#         # if 0.95 <= (angle / 90) and (angle / 90) < 1.05:
#         #     rotated_images.append(RGB_image)
            

#         # if 0.95 <= (angle / 180) and (angle / 90) < 1.05:
#         #     rotated_images.append(RGB_image)
            

#         # if 0.95 <= (angle / 270) and (angle / 90) < 1.05:
#         #     rotated_images.append(RGB_image)

#         fixed_image = imutils.rotate(RGB_image, angle=angle)

#         # plt.imshow(fixed_image)
#         # plt.title(f"After Rotation {idx}")
#         # plt.show()

#         rotated_images.append(fixed_image)

#         rot_file = os.path.join(output_directory, f"rotated_image{idx}.png")
#         cv2.imwrite(rot_file, fixed_image)


#     return rotated_images

@execution_time
def rotate_image(images,boxes,output_directory):
    rot_images=[]
    #angles = []
    for idx,rot_i in enumerate(images):
        RGB_image = rot_i.astype(np.uint8)
        image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2RGB)

        box = boxes[idx][0]
        #print(box)
        tl_x, tl_y = box[0]
        tr_x, tr_y = box[1]

        dx = tr_x - tl_x
        dy = tr_y - tl_y

        angle = np.arctan2(dy, dx)

        angle = np.rad2deg(angle)

        #print(angle)

        fixed_image = imutils.rotate(image, angle=angle)
        rot_images.append(fixed_image)

        plt.imshow(fixed_image)
        plt.show()

        fixed_image_f = cv2.cvtColor(fixed_image, cv2.COLOR_RGB2BGR)

        rot_file = os.path.join(output_directory, f"rotated_image{idx}.png")
        cv2.imwrite(rot_file, fixed_image_f)

    return rot_images









################################################################
#Paddle_ocr
################################################################

def choose_paddle_lang(lang='fr'):
    ocr_model = PaddleOCR(use_angle_cls=True,lang=lang)

    return ocr_model


@execution_time
def Paddle_dec(ocr_model,images):
    B_boxes=[]


    for img in images:
        image = img.astype(np.uint8)
        result = ocr_model.ocr(image, cls=True,rec=False)
        
        # print("here is the result:",result)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = result[0]
        # print(boxes)
        # print("boxes here :", boxes)
        # text = [res[1][0] for res in result[0]]
        # score = [res[1][1] for res in result[0]]
        

        B_boxes.append(boxes)
        # texts.append(text)
        # scores.append(score)

    return B_boxes



@execution_time
def Paddle_rec(ocr_model,images):
    B_boxes=[]
    texts=[]
    scores=[]


    for img in images:
        image = img.astype(np.uint8)
        result = ocr_model.ocr(image, cls=True)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = [res[0] for res in result[0]]

        text = [res[1][0] for res in result[0]]
        score = [res[1][1] for res in result[0]]
        

        B_boxes.append(boxes)
        texts.append(text)
        scores.append(score)

    return B_boxes,texts,scores



def draw_result(images,boxes,text,score,output_directory):

    for idx,img in enumerate(images):
        image = img.astype(np.uint8)

                

        annotated = draw_ocr(image, boxes[idx], text[idx], score[idx], font_path=font_path)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        output_path = os.path.join(output_directory, f"annotated_image_{time()}.png")
        cv2.imwrite(output_path, annotated_bgr)
        plt.imshow(annotated)
        plt.show()


    return 


################################################################
#Count_boxes
################################################################



@execution_time
def count(front_bboxes, bottom_bboxes, text):

    matching_threshold = 50

    relationship_dict = {}
    
    for i, front_bbox in enumerate(front_bboxes):
        front_tl_x, front_tl_y, front_tr_x, front_tr_y = front_bbox
        corresponding_text = tuple(text[i])
        counter = 0
        
        for bottom_box in bottom_bboxes:
            bottom_tl_x, bottom_tl_y, bottom_tr_x, bottom_tr_y = bottom_box
            
            if (
                abs(front_tl_x - bottom_tl_x) < matching_threshold and
                #abs(front_tl_y - bottom_tl_y) < matching_threshold and
                abs(front_tr_x - bottom_tr_x) < matching_threshold 
                #abs(front_tr_y - bottom_tr_y) < matching_threshold
            ):
                counter += 1
        
        relationship_dict[corresponding_text] = counter
        print(relationship_dict)

    for k, v in relationship_dict.items():
        print(f"The number of {k} is {v}") 

    return relationship_dict

from Utils import (
    detect_annotate_dino , segmentation , execution_time,crop_image, sam_predictor,test,Paddle_dec,Paddle_rec,choose_paddle_lang,draw_result,rotate_image,count,#get_text_orientation,
)
import os
import torch

import cv2
from Config import CLASSES
import matplotlib.pyplot as plt


# from PaddleOCR.tools.infer.predict_det import TextDetector


SOURCE_IMAGE_PATH="/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/imagesMedicament/1.png"
SOURCE_IMAGE_PATH2="/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/imagesMedicament/2.png"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pipeline(image_path:str,image_path2:str):
    image = cv2.imread(image_path)
    image2 = cv2.imread(image_path2)

    detections,image_dino,bboxes=detect_annotate_dino(image=image,classes=CLASSES)

    #print('detections',detections)

    #print('front bboxes',bboxes)


    # Save Dino image
    output_dino = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/images_dino"

    filename_dino = os.path.join(output_dino, "Dino_image.png")
    cv2.imwrite(filename_dino, image_dino)

    image_sam=segmentation(detections, sam_predictor, image=image, classes=CLASSES)

    # Save SAM image
    output_dino = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/images_sam"

    filename_SAM = os.path.join(output_dino, "SAM_image.png")
    cv2.imwrite(filename_SAM, image_sam)


    output_crop = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/images_croped"
    os.makedirs(output_crop, exist_ok=True)

    cropped_images=crop_image(detections , output_crop , image)

    output_rot = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/image_rot"
    os.makedirs(output_rot, exist_ok=True)


    model=choose_paddle_lang()



    output_fin = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/images_final"
    os.makedirs(output_fin, exist_ok=True)

    boxes=Paddle_dec(model,cropped_images)

    rot_images=rotate_image(cropped_images,boxes,output_rot)

    boxxes,text,score=Paddle_rec(model,rot_images)

    #print('here is the ocr text : ',text)

    output_fin = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/images_final"
    os.makedirs(output_fin, exist_ok=True)

    draw_result(rot_images,boxxes,text,score,output_fin)


    detections2,bottom_view,bboxes2=detect_annotate_dino(image=image2,classes=CLASSES)

    # Save Dino2 image
    output_dino = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/bottom_img"

    filename_dino = os.path.join(output_dino, "bottom_image.png")
    cv2.imwrite(filename_dino, bottom_view)

    #print('bottom bboxes:',bboxes2)

    count(bboxes,bboxes2,text)















@execution_time
def main():
    image_path = SOURCE_IMAGE_PATH
    image_path2 = SOURCE_IMAGE_PATH2
    pipeline(image_path,image_path2)






if __name__ == "__main__":
    main()

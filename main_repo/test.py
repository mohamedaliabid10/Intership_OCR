import os
import supervision as sv
import torch
import cv2

from groundingdino.util.inference import Model

from typing import List

#torch config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#DINO config
GROUNDING_DINO_CHECKPOINT_PATH = "/home/mohamedaliabid/Desktop/IOv.2/weight/groundingdino_swint_ogc.pth"

#DINO hyperparameter
GROUNDING_DINO_CONFIG_PATH = "/home/mohamedaliabid/Desktop/IOv.2/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CLASSES = ['single medicament boxe']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


#DINO config
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

#add all + s to all classes to detect all object of the same class not only one class
def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]



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

    return detections , annotated_frame


image = cv2.imread("/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/imagesMedicament/2.png")

detections,image_dino=detect_annotate_dino(image=image,classes=CLASSES)
# Save Dino image
output_dino = "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/main_repo/test_img"

filename_dino = os.path.join(output_dino, "Dino_image.png")
cv2.imwrite(filename_dino, image_dino)



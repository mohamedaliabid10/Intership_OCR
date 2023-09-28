import os

#download Dino weights :::: wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
################################################################
#DINO
################################################################

#DINO config
GROUNDING_DINO_CHECKPOINT_PATH = "/home/mohamedaliabid/Desktop/IOv.2/weight/groundingdino_swint_ogc.pth"

#DINO hyperparameter
GROUNDING_DINO_CONFIG_PATH = "/home/mohamedaliabid/Desktop/IOv.2/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CLASSES = ['single medicament boxes']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25


#download sam weight :::: wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
################################################################
#SAM
################################################################

#SAM config
SAM_CHECKPOINT_PATH = "/home/mohamedaliabid/Desktop/IOv.2/sam_models/sam_vit_h_4b8939.pth"
SAM_ENCODER_VERSION = "vit_h"


################################################################
#Paddleocr
################################################################

font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'french.ttf')
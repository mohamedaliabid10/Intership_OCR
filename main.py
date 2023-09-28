#----------------------------------------------------------------
#python -m pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple /// to use CPU
#pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
#git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
#wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
#sudo dpkg -i libssl1.1_1.1.1f-1ubuntu2.19_amd64.deb
#----------------------------------------------------------------

from paddleocr import PaddleOCR, draw_ocr # main OCR dependencies
from matplotlib import pyplot as plt # plot images
import cv2 #opencv
import os # folder directory navigation
import time

ocr_model = PaddleOCR(use_angle_cls=True,lang='fr')

image_directory= "/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/test"
image_files = os.listdir(image_directory)

# Record the start time
start_time = time.time()

for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    output_directory = '/home/mohamedaliabid/cleanedVersion/MedicamentProject-OCR-Approach/test.res'
    img = cv2.imread(image_path)
    result = ocr_model.ocr(img, cls=True)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = [res[0] for res in result[0]]
    text = [res[1][0] for res in result[0]]
    score = [res[1][1] for res in result[0]]
    #you specify the text language displayed on the images here we use french
    font_path = os.path.join('PaddleOCR', 'doc', 'fonts', 'french.ttf')
    annotated = draw_ocr(img, boxes, text, score, font_path=font_path)
    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_directory, image_file)
    cv2.imwrite(output_path, annotated_bgr)
    plt.imshow(annotated)
    plt.show()

# Record the end time
end_time = time.time()

# Calculate the total processing time
total_processing_time = end_time - start_time
print("Total processing time:", total_processing_time, "seconds")


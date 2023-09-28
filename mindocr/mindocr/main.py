import os
import json
from PIL import Image, ImageDraw

image_dir = "/home/mohamedaliabid/IOvisionOCR/image_no_background"

json_file = "/home/mohamedaliabid/IOvisionOCR/mindocr/data.json"

output_dir = "/home/mohamedaliabid/IOvisionOCR/mindocr/images_res"

with open(json_file, "r") as file:
    data = json.load(file)

for image_name, annotations in data.items():
    image_path = os.path.join(image_dir, image_name)

    image = Image.open(image_path)

    draw = ImageDraw.Draw(image)

    for annotation in annotations:
        transcription = annotation["transcription"]
        points = annotation["points"]

        polygon_points = []
        for point in points:
            polygon_points.append((point[0], point[1]))

        draw.polygon(polygon_points, outline="red")

        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)

        draw.text((min_x, min_y - 20), transcription, fill="red")

    output_path = os.path.join(output_dir, image_name)
    image.save(output_path)

    print(f"Bounding boxes saved for {image_name}")

print("Bounding box drawing complete.")

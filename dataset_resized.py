import os
from PIL import Image

input_dir = "dataset"
output_dir = "dataset-resized"
size = (224, 224)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for folder in os.listdir(input_dir):
    in_path = os.path.join(input_dir, folder)
    out_path = os.path.join(output_dir, folder)
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(in_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(in_path, file))
            img = img.resize(size)
            img.save(os.path.join(out_path, file))


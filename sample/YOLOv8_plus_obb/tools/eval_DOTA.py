import json
import os
import argparse

DOTA_CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 
                'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--result_json', type=str, default='../cpp/yolov8_bmcv/results/yolov8s-obb_fp16_1b.bmodel_val_bmcv_cpp_result.json', help='path of input')
args = parser.parse_args()

result_file = args.result_json
output_dir = "TASK1"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for item in os.listdir(output_dir):
    item_path = os.path.join(output_dir, item)
    if os.path.isfile(item_path):
        os.remove(item_path)


with open(result_file, 'r') as file:
    data=json.load(file)

image_names = set()
categorys = {i: [] for i in range(len(DOTA_CLASSES))}
for item in data:
    image_name = os.path.splitext(item["image_name"])[0]
    image_names.add(image_name)
    for res in item['bboxes']:
        box = res["bbox"]
        score = res["score"]
        line_to_write = f"{image_name} {score} "+' '.join(map(str, box))
        categorys[res["category_id"]].append(line_to_write) 

img_name_file = os.path.join(output_dir, "valset.txt")
with open(img_name_file, 'w') as file:
    file.write('\n'.join(image_names)+'\n')


for category_id, lines in categorys.items():
    if lines:
        output_file = os.path.join(output_dir, f"Task1_{DOTA_CLASSES[category_id]}.txt")
        with open(output_file, "w") as file:
            file.write("\n".join(lines)+'\n')

print()
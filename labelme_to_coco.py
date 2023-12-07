import glob
import json
import os

import numpy as np


def convert(labelme_json_dir, coco_json_file, categories):
    labelme_json_files = glob.glob(os.path.join(labelme_json_dir, "*.json"))  # 绝对路径
    coco_json_dict = {'info': None, 'licenses': None, 'type': 'instances',
                      'images': [], 'annotations': [], 'categories': []}

    categories_interchange = dict(zip(categories.values(), categories.keys()))  # 键值互换 {'plane':1,}

    bnd_id = 1  # 框的id(即标注id)，从1开始
    image_id = 0  # 图片id，从0开始
    for json_file in labelme_json_files:  # 循环一次处理一张图片
        with open(json_file, 'r') as f:
            data = json.load(f)  # 加载json文件。字典形式
        image_name = data['imagePath']
        height, width = data['imageHeight'], data['imageWidth']

        images = {
            "id": image_id,
            "file_name": image_name,
            "height": height,
            "width": width
        }
        coco_json_dict["images"].append(images)

        for od in data['shapes']:
            category_id = categories_interchange[od['label']]
            segmentation = [list(np.asarray(od['points']).flatten())]

            # 找出标注点中的外接矩形的四个点
            x_axis = segmentation[0][::2]  # 偶数个是x的坐标
            y_axis = segmentation[0][1::2]  # 奇数个是y的坐标
            # x1,y1 = min(x_axis) - 1,min(y_axis) - 1  # 往外扩展1个像素，也可以不扩展
            x1, y1, x2, y2 = min(x_axis), min(y_axis), max(x_axis), max(y_axis)  # 不扩展
            print(image_name,x1,y1,x2,y2)
            assert x2 > x1 and y2 > y1
            box_width = abs(x2 - x1)
            box_height = abs(y2 - y1)

            ann = {
                "id": bnd_id,  # 同一张图片可能对应多个 ann
                "image_id": image_id,  # 对应的图片ID（与images中的ID对应）
                "category_id": category_id,
                "segmentation": segmentation,
                "area": box_width * box_height,
                "bbox": [x1, y1, box_width, box_height],
                "iscrowd": 0
            }
            coco_json_dict["annotations"].append(ann)
            bnd_id += 1
        image_id += 1

    for cid, cate in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        coco_json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(coco_json_file), exist_ok=True)
    json.dump(coco_json_dict, open(coco_json_file, 'w'), indent=4)  # indent=4为了方便看


if __name__ == '__main__':
    # 使用方法：
    # （1）将训练集或验证集或测试集的标注文件放在某个文件夹下，然后把文件夹的绝对路径赋值给labelme_json_dir
    # （2）指定coco的标注文件的绝对路径，注意不是文件夹
    # （3）自己规定类别映射categories
    labelme_json_dir = r'C:\Users\wqhslf\Desktop\a'
    coco_json_file = r'C:\Users\wqhslf\Desktop\annotations\instances_train2017.json'  # coco标注文件，可以自己建或不建

    categories = {1: 'building'}
    # categories = {1: 'aircraft', 2: 'oiltank', 3: 'overpass', 4: 'playground'}

    convert(labelme_json_dir, coco_json_file, categories)

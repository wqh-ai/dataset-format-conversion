import glob
import json
import os
import shutil

import cv2


def A():
    # 将验证集和其标注文件放在一个文件夹下。剩下的图片标注作为训练集
    txt_file = r'C:\Users\wqhslf\Desktop\LSSSDD\valimgname.txt'
    with open(txt_file, 'r', encoding='utf-8') as f:
        anno_datas = f.readlines()

    for i in anno_datas:
        image_name = i.strip().split('/')[-1]
        txt_name = image_name.replace('.jpg', '.txt')

        image_dir = r'C:\Users\wqhslf\Desktop\LSSSDD\images'
        txt_dir = r'C:\Users\wqhslf\Desktop\LSSSDD\txts'

        new_image_dir = r'C:\Users\wqhslf\Desktop\LSSSDD\val2017'
        new_txt_dir = r'C:\Users\wqhslf\Desktop\LSSSDD\valtxt'

        shutil.move(os.path.join(image_dir, image_name), new_image_dir)
        shutil.move(os.path.join(txt_dir, txt_name), new_txt_dir)


def convert(txt_dir, image_dir, json_file):
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))  # 获取标注文件夹中所有 .txt文件名 存入列表
    json_dict = {'info': None, 'licenses': None, 'type': 'instances',
                 'images': [], 'annotations': [], 'categories': []}

    categories = {1: 'ship'}

    image_id = 0
    bnd_id = 1
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:  # 会自己关闭文件
            anno_datas = f.readlines()

        image_name = txt_file.split('\\')[-1].split('.')[0] + '.jpg'
        image_abs_path = os.path.join(image_dir, image_name)
        # 构造image字段
        image = cv2.imread(image_abs_path)
        height, width, _ = image.shape

        images = {
            "id": image_id,  # 图片的ID编号（每张图片ID是唯一的）
            "file_name": image_name,  # 图片名
            "height": height,
            "width": width
        }
        json_dict["images"].append(images)

        # 构造标注字段
        for data in anno_datas:
            if data is '\n':  # 为了跳过空行
                continue
            # 处理读取的数据，转换成目标格式
            data = data.strip('\n').split(' ')

            category_id = 1
            data = [eval(i) for i in data[1:]]
            cx, cy, w, h = data[0], data[1], data[2], data[3]
            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height

            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                print(image_name, category_id, x1, y1, x2, y2)
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 < 0:
                    x2 = 0
                if y2 < 0:
                    y2 = 0
            box_width, box_height = x2 - x1, y2 - y1

            ann = {
                "id": bnd_id,  # 同一张图片可能对应多个 ann
                "image_id": image_id,  # 对应的图片ID（与images中的ID对应）
                "category_id": category_id,
                "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],
                "area": box_width * box_height,
                "bbox": [x1, y1, box_width, box_height],
                "iscrowd": 0
            }
            json_dict["annotations"].append(ann)
            bnd_id += 1
        image_id += 1

    for cid, cate in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json.dump(json_dict, open(json_file, 'w'), indent=4)


if __name__ == '__main__':
    A()
    # 使用方法：
    # （1）将训练集或验证集或测试集的标注文件所在文件夹的绝对路径赋值给txt_dir
    # （2）将训练集或验证集或测试集的图片所在文件夹的绝对路径赋值给image_dir
    # （3）创建annotations文件夹以及instances_train2017.json，instances_val2017.json文件，
    #     然后把json文件的绝对路径赋值给json_file。可以不用创建json文件和annotations文件夹，代码会创建

    txt_dir = r'C:\Users\wqhslf\Desktop\LSSSDD\traintxt'
    image_dir = r'C:\Users\wqhslf\Desktop\LSSSDD\train2017'
    json_file = r'C:\Users\wqhslf\Desktop\LSSSDD\annotations\instances_train2017.json'

    # convert(txt_dir, image_dir, json_file)

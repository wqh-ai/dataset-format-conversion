import glob
import json
import os
import random
import shutil

import cv2


# VisDrone数据标注格式，从左到右依次是:
# <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
# 标注中，类别为数字，可以自己映射，为了保持一致，还是要注意。


def B():
    # 根据标注信息中的框的第一个类别，统计每张图片的类别信息和该类别的数量
    dict1 = {0: 0,
             1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
             6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
             11: 0}
    # 统计该类图片的名字
    dict2 = {0: [],
             1: [], 2: [], 3: [], 4: [], 5: [],
             6: [], 7: [], 8: [], 9: [], 10: [],
             11: []}

    txt_dir = r'E:\cocoData\VisDrone\traintxt'
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:  # 会自己关闭文件
            anno_datas = f.readlines()

        data = anno_datas[0]
        data = data.strip('\n').split(',')

        class_id = eval(data[-3])
        dict1[class_id] += 1  # 类别统计

        image_name = txt_file.split('\\')[-1].split('.')[0] + '.jpg'
        dict2[class_id].append(image_name)  # 图片统计

    for k, v in dict1.items():
        print(k, '\t', v)

    # 将每个类别对用的图片名字写入txt文件
    out_txt_dir = r'E:\cocoData\VisDrone'
    for i in range(len(dict1)):
        with open(os.path.join(out_txt_dir, f'category_{i}.txt'), 'w') as f:
            for j in dict2[i]:
                f.writelines(j + '\n')


def C():
    # 构造valimgname.txt文件。
    # 读取单类别的txt文件，category_*.txt，然后随机选择一定数量的图像作为验证集
    txt_dir = r'E:\cocoData\VisDrone'  # 该路径也是输出文件的路径
    cate_nums = 12  # 需要自己指定
    rlist = []  # 存放验证集的图片名
    for i in range(cate_nums):
        txt_file = os.path.join(txt_dir, f'category_{i}.txt')
        with open(txt_file, 'r', encoding='utf-8') as f:  # 会自己关闭文件
            datas = f.readlines()  # 每行数据会有换行符

        n = round(len(datas) * 0.2)  # 计算验证集的数量，四舍五入取整
        rlist += random.sample(datas, n)  # 不重复选取

    with open(os.path.join(txt_dir, 'valimgname.txt'), 'w') as f:
        for j in rlist:
            f.writelines(j)  # 字符串j带有换行符


def D():
    # 将验证集和其标注文件放在一个文件夹下。剩下的图片标注作为训练集
    txt_file = r'E:\cocoData\VisDrone\valimgname.txt'
    with open(txt_file, 'r', encoding='utf-8') as f:
        anno_datas = f.readlines()

    for i in anno_datas:
        image_name = i.strip('\n')
        txt_name = image_name.replace('.jpg', '.txt')

        image_dir = r'E:\cocoData\VisDrone\train2017'
        txt_dir = r'E:\cocoData\VisDrone\traintxt'

        new_image_dir = r'E:\cocoData\VisDrone\val2017'
        new_txt_dir = r'E:\cocoData\VisDrone\valtxt'

        shutil.move(os.path.join(image_dir, image_name), new_image_dir)
        shutil.move(os.path.join(txt_dir, txt_name), new_txt_dir)


def convert(txt_dir, image_dir, json_file):
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))  # 获取标注文件夹中所有 .txt文件名 存入列表
    json_dict = {'info': None, 'licenses': None, 'type': 'instances',
                 'images': [], 'annotations': [], 'categories': []}

    categories = {0: 'ignored regions',
                  1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
                  6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor',
                  11: 'others'}

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
            data = data.strip('\n').split(',')
            category_id = eval(data[-3])
            data = [eval(i) for i in data[:4]]
            x1, y1, box_width, box_height = data[0], data[1], data[2], data[3]
            assert x1 >= 0 and y1 >= 0, f'{image_name}'

            ann = {
                "id": bnd_id,  # 同一张图片可能对应多个 ann
                "image_id": image_id,  # 对应的图片ID（与images中的ID对应）
                "category_id": category_id,
                "segmentation": [[x1, y1, x1, y1 + box_height, x1 + box_width, y1 + box_height, x1 + box_width, y1]],
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
    # B()
    # C()
    # D()
    # 使用方法：
    # （1）将训练集或验证集或测试集的标注文件所在文件夹的绝对路径赋值给txt_dir
    # （2）将训练集或验证集或测试集的图片所在文件夹的绝对路径赋值给image_dir
    # （3）创建annotations文件夹以及instances_train2017.json，instances_val2017.json文件，
    #     然后把json文件的绝对路径赋值给json_file。
    #     可以不用创建json文件和annotations文件夹，代码会创建

    txt_dir = r'E:\cocoData\VisDrone\valtxt'
    image_dir = r'E:\cocoData\VisDrone\val2017'
    json_file = r'E:\cocoData\VisDrone\annotations\instances_val2017.json'

    convert(txt_dir, image_dir, json_file)

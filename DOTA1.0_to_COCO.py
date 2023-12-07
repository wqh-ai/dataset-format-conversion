#  DOTA1.0格式：
# 'imagesource':imagesource
# 'gsd':gsd
# x 1, y 1, x 2, y 2, x 3, y 3, x 4, y 4, category, difficult
# x 1, y 1, x 2, y 2, x 3, y 3, x 4, y 4, category, difficult
# category：目标所属类别；
# diffcult：该目标实例的检测难度，1为高，0为低。

# 飞机，轮船，储罐，棒球场，网球场，篮球场，地面跑道，港口，桥梁，大型车辆，小型车辆，直升机，环形交叉路口，足球场和篮球场
import glob
import json
import os
import shutil

import cv2


def A():
    # 将标注文件中没有目标的标注和对应的图片移动到noobj文件夹
    txt_dir = r'C:\Users\wqhslf\Desktop\a'
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:  # 会自己关闭文件
            anno_datas = f.readlines()

        if anno_datas:
            continue
        else:
            print(txt_file)
            # 将txt标注文件剪切粘贴到另一个文件夹
            out_dir = r'C:\Users\wqhslf\Desktop\noobj'
            shutil.move(txt_file, out_dir)

            image_dir = r'C:\Users\wqhslf\Desktop\a'
            shutil.move(os.path.join(image_dir, txt_file.split('\\')[-1].replace('.txt', '.jpg')), out_dir)


def B():
    # 根据标注信息中的框的第一个类别，统计每张图片的类别信息和该类别的数量
    dict1 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
             6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
             11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
    # 统计该类图片的名字
    dict2 = {1: [], 2: [], 3: [], 4: [], 5: [],
             6: [], 7: [], 8: [], 9: [], 10: [],
             11: [], 12: [], 13: [], 14: [], 15: []}

    categories = {1: 'plane', 2: 'baseball-diamond', 3: 'bridge', 4: 'ground-track-field', 5: 'small-vehicle',
                  6: 'large-vehicle', 7: 'ship', 8: 'tennis-court', 9: 'basketball-court', 10: 'storage-tank',
                  11: 'soccer-ball-field', 12: 'roundabout', 13: 'harbor', 14: 'swimming-pool', 15: 'helicopter'}

    categories_interchange = dict(zip(categories.values(), categories.keys()))  # 键值互换 {'plane':1,}

    txt_dir = r'E:\cocoData\DOTA1.0\labelTxt'
    txt_files = glob.glob(os.path.join(txt_dir, "*.txt"))
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:  # 会自己关闭文件
            anno_datas = f.readlines()

        data = anno_datas[0]
        data = data.strip('\n').split(' ')

        class_id = categories_interchange[data[-2]]
        dict1[class_id] += 1  # 类别统计

        image_name = txt_file.split('\\')[-1].split('.')[0] + '.jpg'
        dict2[class_id].append(image_name)  # 图片统计

    for k, v in dict1.items():
        print(k, '\t', v)

    print('-' * 20)

    for k, v in dict2.items():
        print(k)
        for i in v:
            print(i)


def C():
    # 将验证集和其标注文件放在一个文件夹下。剩下的图片标注作为训练集
    txt_file = r'E:\cocoData\DOTA1.0\valimgname.txt'
    with open(txt_file, 'r', encoding='utf-8') as f:
        anno_datas = f.readlines()

    for i in anno_datas:
        image_name = i.strip('\n')
        txt_name = image_name.replace('.jpg', '.txt')

        image_dir = r'E:\cocoData\DOTA1.0\images'
        txt_dir = r'E:\cocoData\DOTA1.0\labelTxt'

        new_image_dir = r'E:\cocoData\DOTA1.0\val2017'
        new_txt_dir = r'E:\cocoData\DOTA1.0\valtxt'

        shutil.move(os.path.join(image_dir, image_name), new_image_dir)
        shutil.move(os.path.join(txt_dir, txt_name), new_txt_dir)


def convert(txt_dir, image_dir, json_file):
    txt_files = glob.glob(os.path.join(txt_dir, '*.txt'))  # 获取标注文件夹中所有 .txt文件名 存入列表
    json_dict = {'info': None, 'licenses': None, 'type': 'instances',
                 'images': [], 'annotations': [], 'categories': []}

    categories = {1: 'plane', 2: 'baseball-diamond', 3: 'bridge', 4: 'ground-track-field', 5: 'small-vehicle',
                  6: 'large-vehicle', 7: 'ship', 8: 'tennis-court', 9: 'basketball-court', 10: 'storage-tank',
                  11: 'soccer-ball-field', 12: 'roundabout', 13: 'harbor', 14: 'swimming-pool', 15: 'helicopter'}

    categories_interchange = dict(zip(categories.values(), categories.keys()))  # 键值互换 {'plane':1,}

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
            # 此时data的格式
            # ['937.0', '913.0', '921.0', '912.0', '923.0', '874.0', '940.0', '875.0',
            # 'small-vehicle', '0'] <class 'list'>
            # 标注文件一行为一个字符串，即data为一个字符串
            category_id = categories_interchange[data[-2]]
            data = [eval(i) for i in data[:-2]]  # 此时data为列表，列表元素为四个旋转框的坐标值
            x = data[::2]  # 得到四个点的x坐标
            y = data[1::2]  # 得到四个点的y坐标
            x1, y1, x2, y2 = min(x), min(y), max(x), max(y)  # 旋转框变横框
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or \
                    x1 > width or y1 > height or x2 > width or y2 > height or \
                    x1 >= x2 or y1 >= y2:
                print(image_name, category_id, x1, y1, x2, y2, width, height)
                if x1 < 0:
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if x2 < 0:
                    x2 = 0
                if y2 < 0:
                    y2 = 0
                if x1 > width:
                    x1 = width
                if y1 > height:
                    y1 = height
                if x2 > width:
                    x2 = width
                if y2 > height:
                    y2 = height
                if x1 >= x2:
                    x1 = x2 - 1
                if y1 >= y2:
                    y1 = y2 - 1
            box_width, box_height = x2 - x1, y2 - y1

            ann = {
                "id": bnd_id,  # 同一张图片可能对应多个 ann
                "image_id": image_id,  # 对应的图片ID（与images中的ID对应）
                "category_id": category_id,
                "segmentation": [[x1, y1, x1, y2, x2, y2, x2, y1]],  # 用旋转框的标注作为分割点
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
    # B()
    # C()
    # DOTA1.0旋转框转换coco代码。
    # 使用方法：
    # （1）将训练集或验证集或测试集的标注文件所在文件夹的绝对路径赋值给txt_dir
    # （2）将训练集或验证集或测试集的图片所在文件夹的绝对路径赋值给image_dir
    # （3）创建annotations文件夹以及instances_train2017.json，instances_val2017.json文件，
    #     然后把json文件的绝对路径赋值给json_file。可以不用创建json文件和annotations文件夹，代码会创建

    txt_dir = r'E:\cocoData\DOTA1.0\traintxt'
    image_dir = r'E:\cocoData\DOTA1.0\train2017'
    json_file = r'E:\cocoData\DOTA1.0\annotations\instances_train2017.json'

    # convert(txt_dir, image_dir, json_file)

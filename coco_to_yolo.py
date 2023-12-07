import json
import os


# YoLo数据文件结构
# ├─test
# │  ├─images
# │  └─labels
# ├─train
# │  ├─images
# │  └─labels
# └─valid
#     ├─images
#     └─labels

# x1,y1,w,h coco框的格式
# yolo的标注文件内容
# <object-class> <x> <y> <width> <height> == [x_center, y_center, w, h](做归一化)
# <object-class>：对象的标签索引x，y：目标的中心坐标，相对于图片的H和W做归一化。即x/W，y/H。
# width，height：目标（bbox）的宽和高，相对于图像的H和W做归一化。

def A():
    # txt文件的写操作
    data = ['a', 'b', 'c', 1]
    # 单层列表写入文件
    with open("data.txt", "w") as f:
        # for i in data:
        #     i += '\n'
        #     f.writelines(i)
        f.writelines(data)


def B():
    # txt文件读取操作
    with open("data.txt", 'r', encoding='utf-8') as f:  # 会自己关闭文件
        anno_datas = f.readlines()
    print(anno_datas)


def convert(coco_json_file, yolo_txt_out_dir):
    with open(coco_json_file, 'r') as f:
        data = json.load(f)  # 加载json文件。字典形式

    categories = {}
    for cate_dict in data['categories']:
        categories[cate_dict['id']] = cate_dict['name']

    images = data['images']  # 获取图片的信息
    annotations = data['annotations']  # 获取所有标注的信息

    for i in range(len(images) - 1, -1, -1):  # image为字典
        image_id = images[i]['id']  # 通过图片的id拿到框的信息
        image_name = images[i]['file_name']  # 用于构造标注的名字
        image_height = images[i]['height']  # 用于归一化
        image_width = images[i]['width']  # 用于归一化

        obj_list = []
        for j in range(len(annotations) - 1, -1, -1):  # 逆序循环处理一张图上的多个目标
            anno = annotations[j]
            if anno['image_id'] == image_id:
                category_id = anno['category_id']
                x1, y1, box_width, box_height = anno['bbox']  # coco格式的形式

                x_center = (x1 + box_width / 2) / image_width
                y_center = (y1 + box_height / 2) / image_height
                w = box_width / image_width
                h = box_width / image_height

                obj_list.insert(0, [category_id, x_center, y_center, w, h])  # 因为是倒序循环图片中的目标，所以在列表开头插入

                annotations.pop(j)
            else:
                break

        # 保存txt文件，自动打开并创建
        txt_name = image_name.split('.')[0] + '.txt'
        with open(os.path.join(yolo_txt_out_dir, txt_name), 'w') as f:
            for obj in obj_list:
                r = ''
                for o in obj:
                    r += str(o) + ' '
                f.writelines(r + '\n')

        images.pop(i)

    # 保存类别映射
    with open(os.path.join(yolo_txt_out_dir, 'classes.txt'), 'w') as f:
        for k, v in categories.items():
            r = str(k) + ' ' + v
            f.writelines(r + '\n')


if __name__ == '__main__':
    # A()
    # B()
    # 使用说明：
    # （1）将coco格式的json文件的文件名以及绝对路径赋值给coco_json_file
    # （2）指定yolo格式标注的输出文件夹的绝对路径
    # （3）注意：会多一个类别映射文件夹classes.txt。yolo数据集的文件结需要自己创建，本代码只负责转换

    coco_json_file = r'E:\cocoData\mcoco\annotations2.0\instances_val2017.json'
    yolo_txt_out_dir = r'C:\Users\wqhslf\Desktop\b'
    convert(coco_json_file, yolo_txt_out_dir)

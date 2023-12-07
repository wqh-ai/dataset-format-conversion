import glob
import os
import shutil

import cv2


def overlap(box1, box2):
    # 问题定义：给定两个边与坐标轴平行的矩形，分别由左上角与右下角两点指定，即矩形(P1，P2)与(P3，P4)，判断两矩形是否相交。
    #
    # 我的思路：如下图所示，首先求出P1与P3点在X方向较大值与Y方向较大值的交点，在下图中就是P3，用红点(记为M点)表示。
    #
    # 然后求出P2与P4点在X方向较小值与Y方向较小值的交点，在下图中就是P2，用橙色点(记为N点)表示。
    #
    # 如果M点的X坐标和Y坐标值均比N点相应的X坐标和Y坐标值小，亦即M和N可以分别构成一个矩形的左上角点和右上角点，
    # 则两矩形相交；其余情况则不相交。

    # 判断两个矩形是否相交
    # 思路来源于:https://www.cnblogs.com/avril/archive/2013/04/01/2993875.html
    # 然后把思路写成了代码
    minx1, miny1, maxx1, maxy1 = box1
    minx2, miny2, maxx2, maxy2 = box2
    minx = max(minx1, minx2)
    miny = max(miny1, miny2)
    maxx = min(maxx1, maxx2)
    maxy = min(maxy1, maxy2)
    if minx > maxx or miny > maxy:  # 好像是错的
        return False
    else:
        return True


def split_anno(p: list, txt_file, anno_out_dir, new_img_name: str):
    with open(txt_file, 'r', encoding='utf-8') as f:  # 会自己关闭文件
        anno_datas = f.readlines()

    rlist = []
    for data in anno_datas[2:]:  # DOTA1.5为anno_datas[2:]，DOTA1.0为anno_datas
        if data is '\n':
            continue
        data = data.strip('\n').split(' ')

        category_name = data[-2]
        data = [eval(i) for i in data[:-2]]  # 此时data为列表
        x = data[::2]  # 得到四个点的x坐标
        y = data[1::2]  # 得到四个点的y坐标
        x1, y1, x2, y2 = min(x), min(y), max(x), max(y)

        # 判断是否相交，并求交点
        mx = max(p[0], x1)  # 注意：找最大
        my = max(p[1], y1)
        nx = min(p[2], x2)  # 注意：找最小
        ny = min(p[3], y2)
        if mx < nx and my < ny:
            # 原图上的坐标映射到新图
            rlist.append([mx - p[0], my - p[1], nx - p[0], ny - p[1], category_name])

    new_txt_name = new_img_name.split('.')[0] + '.txt'
    with open(os.path.join(anno_out_dir, new_txt_name), 'w') as f:
        for i in rlist:
            r = ''
            for j in i:
                r += str(j) + ' '
            f.writelines(r + '0' + '\n')  # 新txt标注保持与原始的一样，方便后面转换


def split(img_dir, img_format, txt_dir, new_size: tuple, new_img_out_dir):
    img_paths = glob.glob(os.path.join(img_dir, f'*.{img_format}'))
    size_h, size_w = new_size
    for img_pth in img_paths:
        image = cv2.imread(img_pth)
        height, width, _ = image.shape

        m = height // size_h + 1
        n = width // size_w + 1

        x1, y1 = -size_w, -size_h
        x2 = y2 = 0
        plist = []  # 存放描述一个区域的两个点
        for i in range(m):
            if m == n == 1:  # 图像的宽高都不够切割，直接不切割，将其直接移动到新图片输出文件夹
                plist.append([0, 0, height, width])
                shutil.copy(img_pth, new_img_out_dir)
                shutil.copy(os.path.join(txt_dir, img_pth.split('\\')[-1].split('.')[0] + '.txt'),
                            new_img_out_dir)  # 移动标注文件
                break

            y2 += size_h
            y1 += size_h
            if y2 > height:
                y1 = y1 - (y2 - height)
                y2 = height
            if y1 < 0:  # 图像高度不够窗口大小就按原图的高作为窗口的高，这样就不用进行填充
                y1 = 0

            for j in range(n):
                x2 += size_w
                x1 += size_w
                if x2 > width:
                    x1 = x1 - (x2 - width)
                    x2 = width
                if x1 < 0:  # 图像宽度不够窗口大小就按原图的宽作为窗口的宽，这样就不用进行填充
                    x1 = 0

                plist.append([x1, y1, x2, y2])

                crop = image[y1:y2, x1:x2]
                new_img_name = img_pth.split('\\')[-1].split('.')[0] + f'_{x1}_{y1}_{x2}_{y2}.{img_format}'
                cv2.imwrite(os.path.join(new_img_out_dir, new_img_name), crop)

                # 切割txt标注
                split_anno([x1, y1, x2, y2],
                           os.path.join(txt_dir, img_pth.split('\\')[-1].split('.')[0] + '.txt'),
                           new_img_out_dir, new_img_name)

                if x2 == width:  # 处理完一行，横坐标初始化
                    x1 = -size_w
                    x2 = 0

        print(len(plist), plist)


if __name__ == '__main__':
    # 使用方法：
    # （1）将需要切割的图片的目录赋值给img_dir
    # （2）将需要切割的图片的后缀名赋值给img_format，不用打点
    # （3）将需要切割的图片对应的标注文件的目录赋值给txt_dir
    # （4）将需要的新图像的尺寸赋值给new_size
    # （5）指定新图像的输出路径，赋值给new_img_out_dir
    #  ---------注意标注文件--------
    img_dir = r'E:\cocoData\DOTA1.0_1.5\valimages'
    img_format = 'jpg'
    txt_dir = r'E:\cocoData\DOTA1.0_1.5\valtxt1.5'
    new_size = (640, 640)  # h,w
    new_img_out_dir = r'C:\Users\wqhslf\Desktop\a'

    split(img_dir, img_format, txt_dir, new_size, new_img_out_dir)

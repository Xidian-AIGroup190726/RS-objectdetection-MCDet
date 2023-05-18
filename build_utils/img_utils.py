import numpy as np
import cv2
from PIL import Image

def letterbox_image(image, size=(416,416)):

    im = np.array(image)
    ih,iw,_ = im.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128)) # 初始化一张416*416大小的三通道灰白图像
    #new_image.show()
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
   # new_image.save("./tupian.jpg", quality=95)
    return new_image

def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img: 输入的图像numpy格式
    :param new_shape: 输入网络的shape
    :param color: padding用什么颜色填充
    :param auto:
    :param scale_fill: 简单粗暴缩放到指定大小
    :param scale_up:  只缩小，不放大
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[0] - new_unpad[0]
    dh = new_shape[1] - new_unpad[1]
    #dw, dh = int(new_shape[1]) - int(new_unpad[0]), int(new_shape[0]) - int(new_unpad[1])  # wh padding
    #if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
    #    # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
    #    dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    #elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
    #    dw, dh = 0, 0
    #    new_unpad = new_shape
    #    ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    #cv2.namedWindow("image")  # 创建一个image的窗口
    #cv2.imshow("image", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img, ratio, (dw, dh)









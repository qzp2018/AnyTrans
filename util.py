import datetime
import os
import cv2
import numpy as np
from PIL import Image
import math
# from simple_lama_inpainting import SimpleLama
# simple_lama = SimpleLama() 

def image_crop_new(boxes,image_path,idx):
    pil_image=cv2.imread(image_path)
    min_x = max(min([x for x, _ in boxes]),0)
    max_x = max([x for x, _ in boxes])
    min_y = max(min([y for _, y in boxes]),0)
    max_y = max([y for _, y in boxes])
    patch_image = pil_image[min_y:max_y, min_x:max_x]
    if image_path[-10:-4]=='reedit':
        save_image_path= image_path[:-11]+f'{idx}crop.png'
    else:
        save_image_path= image_path[:-4]+f'{idx}crop.png'
    cv2.imwrite(save_image_path,patch_image)
    return save_image_path



def image_crop(pil_imgae,boxes,image_path,idx):
    pil_image=cv2.imread(image_path)
    min_x = max(min([x for x, _ in boxes]),0)
    max_x = max([x for x, _ in boxes])
    min_y = max(min([y for _, y in boxes]),0)
    max_y = max([y for _, y in boxes])
    patch_image = pil_image[min_y:max_y, min_x:max_x]
    if image_path[-10:-4]=='reedit':
        save_image_path= image_path[:-11]+f'{idx}crop.png'
    else:
        save_image_path= image_path[:-4]+f'{idx}crop.png'
    cv2.imwrite(save_image_path,patch_image)
    return save_image_path


def image_crop_google(boxes,image_path,idx):
    pil_image=cv2.imread(image_path)
    min_x = max(min([x for x, _ in boxes]),0)
    max_x = max([x for x, _ in boxes])
    min_y = max(min([y for _, y in boxes]),0)
    max_y = max([y for _, y in boxes])
    patch_image = pil_image[min_y:max_y, min_x:max_x]
    save_image_path= image_path[:-4]+f'{idx}crop.png'
    cv2.imwrite(save_image_path,patch_image)
    return save_image_path

def save_images(img_list, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    folder_path = os.path.join(folder, date_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    time_str = now.strftime("%H_%M_%S")
    for idx, img in enumerate(img_list):
        image_number = idx + 1
        filename = f"{time_str}_{image_number}.jpg"
        save_path = os.path.join(folder_path, filename)
        cv2.imwrite(save_path, img[..., ::-1])


def check_channels(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels > 3:
        image = image[:, :, :3]
    return image


def resize_image(img, max_length=768):
    height, width = img.shape[:2]
    max_dimension = max(height, width)

    if max_dimension > max_length:
        scale_factor = max_length / max_dimension
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    return img


def resize_image_boxes(img,boxes, max_length=768):
    height, width = img.shape[:2]
    height_original,width_original=height,width
    max_dimension = max(height, width)
    if max_dimension > max_length:
        scale_factor = max_length / max_dimension
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    height_end,width_end=img.shape[:2]
    height_scale=height_end/height_original
    width_scale=width_end/width_original

    new_boxes=[]
    for box_coordinates in boxes:
        box_coordinates=np.array(box_coordinates, dtype=np.int32)
        rect = cv2.minAreaRect(box_coordinates)
        center, size, angle = rect
        if angle<45:
            # 调整box
            size_new=(size[0]*width_scale,size[1]*height_scale)    
        else:
            size_new=(size[0]*height_scale,size[1]*width_scale) 

        # 调整中心点坐标
        new_center = (center[0] * width_scale, center[1] * height_scale)          
        new_rect = (new_center, size_new, angle)
        new_box_coordinates = (cv2.boxPoints(new_rect).tolist())
        new_coords = []
        for i, coord in enumerate(new_box_coordinates):
            if i in (0, 1):  # 对于检测框的左上角和左下角进行向下取整
                new_coords.append([math.floor(x) for x in coord])
            elif i in (2, 3):  # 对于检测框的右下角和右上角进行向上取整
                new_coords.append([math.ceil(x) for x in coord])
        new_boxes.append(new_coords)

    return img,new_boxes




# 计算连续两点间的距离
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def enlarge_box_bigger(box):
    # 找到最小外接矩形
    width=distance(box[0],box[1])
    height=distance(box[2],box[3])  
    delta_width=width*0.06
    delta_height=height*0.06
    # 计算矩形的中心、宽度、高度和旋转角
    box=np.array(box, dtype=np.int32)
    rect = cv2.minAreaRect(box)
    center, size, angle = rect
    if angle<45:
        size_new=(size[0]+delta_width,size[1]+delta_height)
    else:
        size_new=(size[0]+delta_height,size[1]+delta_width)
    # 生成新的最小外接矩形的四个顶点
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    # # 获取图像大小
    # img_height,img_width = pil_image.shape[:2]
    # # 确保放大后的box坐标不超出图像边界
    # new_box[:, 0] = np.clip(new_box[:, 0], 1, img_width-10)
    # new_box[:, 1] = np.clip(new_box[:, 1], 1, img_height-10)
    return new_box


def resize_mask2(pil_image,box_coordinates):
    # 获取图像大小
    height,width = pil_image.shape[:2]
    # 将坐标数据转换为float类型
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    whether_erase=False

    dis_kuan=np.linalg.norm(box_coordinates[1]-box_coordinates[0])
    dis_gao=np.linalg.norm(box_coordinates[2]-box_coordinates[1])       
    # 找到最小外接矩形
    rect = cv2.minAreaRect(box_coordinates)
    # 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect
    scale_factor=word_count_new/word_count_old
    if mode=='ch2en':
        scale_factor=scale_factor/2.5
    elif mode=='en2ch' or mode=='fr2ch':
        scale_factor=scale_factor*1.8 
    else:
        scale_factor=scale_factor
    if scale_factor>=0.8 and scale_factor<=1.2:
        scale_factor=1
    
    if scale_factor<0.8:#尺度缩小过大才用擦图
        whether_erase=True
    if angle<45:
        # 根据单词数翻译前后的比例调整矩形的长度
        size_new=(size[0]*scale_factor,size[1])    
    else:
        size_new=(size[0],size[1]*scale_factor)    
    # 生成新的最小外接矩形的四个顶点
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    new_box = np.int0(new_box)
    # 创建一个全黑的图像作为初始掩码
    mask = np.zeros((height,width), dtype=np.uint8) 
    # 在掩码上绘制新的多边形，使用白色填充
    cv2.drawContours(mask, [new_box], 0, (255), -1)
    mask = 255 - mask
    return mask,whether_erase




def resize_mask(img_path,box_coordinates, word_count_old,word_count_new,mode):
    pil_image=cv2.imread(img_path)
    # 获取图像大小
    height,width = pil_image.shape[:2]
    # 将坐标数据转换为float类型
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    whether_erase=False

    dis_kuan=np.linalg.norm(box_coordinates[1]-box_coordinates[0])
    dis_gao=np.linalg.norm(box_coordinates[2]-box_coordinates[1])       
    # 找到最小外接矩形
    rect = cv2.minAreaRect(box_coordinates)
    # 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect
    scale_factor=word_count_new/word_count_old
    if mode=='ch2en':
        scale_factor=scale_factor/2.5
    elif mode=='en2ch' or mode=='fr2ch':
        scale_factor=scale_factor*1.8 
    else:
        scale_factor=scale_factor
    if scale_factor>=0.8 and scale_factor<=1.2:
        scale_factor=1
    
    if scale_factor<0.8:#尺度缩小过大才用擦图
        whether_erase=True
    if angle<45:
        # 根据单词数翻译前后的比例调整矩形的长度
        size_new=(size[0]*scale_factor,size[1])    
    else:
        size_new=(size[0],size[1]*scale_factor)    
    # 生成新的最小外接矩形的四个顶点
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    new_box = np.int0(new_box)
    # 创建一个全黑的图像作为初始掩码
    mask = np.zeros((height,width), dtype=np.uint8) 
    # 在掩码上绘制新的多边形，使用白色填充
    cv2.drawContours(mask, [new_box], 0, (255), -1)
    mask = 255 - mask
    return mask,whether_erase

def resize_mask_returnbox(img_path,box_coordinates, word_count_old,word_count_new,mode):
    pil_image=cv2.imread(img_path)
    # 获取图像大小
    height,width = pil_image.shape[:2]
    # 将坐标数据转换为float类型
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    whether_erase=False

    dis_kuan=np.linalg.norm(box_coordinates[1]-box_coordinates[0])
    dis_gao=np.linalg.norm(box_coordinates[2]-box_coordinates[1])       
    # 找到最小外接矩形
    rect = cv2.minAreaRect(box_coordinates)
    # 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect
    scale_factor=word_count_new/word_count_old
    if mode=='ch2en':
        scale_factor=scale_factor/3
    elif mode=='en2ch' or mode=='fr2ch':
        scale_factor=scale_factor*1.8 
    else:
        scale_factor=scale_factor
    if scale_factor>=0.8 and scale_factor<=1.2:
        scale_factor=1
    
    if scale_factor<0.8:#尺度缩小过大才用擦图
        whether_erase=True
    if angle<45:
        # 根据单词数翻译前后的比例调整矩形的长度
        size_new=(size[0]*scale_factor,size[1])    
    else:
        size_new=(size[0],size[1]*scale_factor)    
    # 生成新的最小外接矩形的四个顶点
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    new_box = np.int0(new_box)
    # 创建一个全黑的图像作为初始掩码
    mask = np.zeros((height,width), dtype=np.uint8) 
    # 在掩码上绘制新的多边形，使用白色填充
    cv2.drawContours(mask, [new_box], 0, (255), -1)
    mask = 255 - mask
    return mask,whether_erase,new_box

def resize_mask_returnbox_suokuan(img_path,box_coordinates, word_count_old,word_count_new,mode):
    pil_image=cv2.imread(img_path)
    # 获取图像大小
    height,width = pil_image.shape[:2]
    # 将坐标数据转换为float类型
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    whether_erase=False

    dis_kuan=np.linalg.norm(box_coordinates[1]-box_coordinates[0])
    dis_gao=np.linalg.norm(box_coordinates[2]-box_coordinates[1])       
    # 找到最小外接矩形
    rect = cv2.minAreaRect(box_coordinates)
    # 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect
    scale_factor=word_count_new/word_count_old
    if mode=='ch2en':
        scale_factor=scale_factor/2.9

    elif mode=='en2ch' or mode=='fr2ch':
        scale_factor=scale_factor*1.8
    elif mode=='en2jp':
        scale_factor=scale_factor/1.2
    else:
        scale_factor=scale_factor

    if scale_factor>=0.8 and scale_factor<=1.2:
        scale_factor=1
        size_new=(size[0],size[1])
    if scale_factor<0.8 and mode=='ch2en':
        scale_factor=1
        size_new=(size[0],size[1])
    if scale_factor<0.8 and mode!='ch2en':#尺度缩小用擦图
        whether_erase=True
        if angle<45:
            # 根据单词数翻译前后的比例调整矩形的长度
            size_new=(size[0]*scale_factor,size[1])    
        else:
            size_new=(size[0],size[1]*scale_factor)

    if scale_factor>1.2:#尺度过大则用缩宽
        whether_erase=True
        if angle<45:
            # 根据单词数翻译前后的比例调整矩形的长度
            size_new=(size[0],size[1]/scale_factor)    
        else:
            size_new=(size[0]/scale_factor,size[1])

    # 生成新的最小外接矩形的四个顶点
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    new_box = np.int0(new_box)
    # 创建一个全黑的图像作为初始掩码
    mask = np.zeros((height,width), dtype=np.uint8) 
    # 在掩码上绘制新的多边形，使用白色填充
    cv2.drawContours(mask, [new_box], 0, (255), -1)
    mask = 255 - mask
    return mask,whether_erase,new_box



def pad_mask_styletext(img_path,box_coordinates,edited_img_path):
    # 读取原图和编辑后的图片
    ori_image = cv2.imread(img_path)
    edited_image = cv2.imread(edited_img_path)
    
    # 获取原图中需要替换的区域的检测框坐标
    pts_dst = np.array(box_coordinates, dtype='float32')

    # 获取编辑后图片的尺寸
    h, w = edited_image.shape[:2]
    # 调整pts_src的顺序以避免旋转
    pts_src = np.array([[0, h], [0, 1], [w-1, 1], [w-1, h]], dtype='float32')
    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    
    # 进行变换，将编辑后图像适配到原图的检测框中
    transformed_edited_image = cv2.warpPerspective(edited_image, M, (ori_image.shape[1], ori_image.shape[0]))

    # 创建一个掩模，代表替换区域
    mask = np.zeros(ori_image.shape, dtype='uint8')
    cv2.fillPoly(mask, [np.int32(pts_dst)], (255, 255, 255))

    # 使用掩模将原图中替换区域部分设为0
    ori_image = cv2.bitwise_and(ori_image, cv2.bitwise_not(mask))
    
    # 使用掩模将变换后的编辑图像的替换部分加入到原图
    result_image = cv2.add(ori_image, cv2.bitwise_and(transformed_edited_image, mask))
    
    if 'inpainted' not in (img_path.split('/'))[-1]:
        image_name=(img_path.split('/'))[-1][:-4]+'_inpainted.png'
    else:
        image_name=img_path
    save_path= os.path.dirname(img_path)
    fused_image_path=os.path.join(save_path,f'{image_name}')

    # transformed_edited_image_path=os.path.join(save_path,f'trans_{image_name}')
    # cv2.imwrite(transformed_edited_image_path,transformed_edited_image)
    cv2.imwrite(fused_image_path, result_image)
    return fused_image_path





def resize_mask_returnbox_suokuan_woresize(img_path,box_coordinates, word_count_old,word_count_new,mode):
    pil_image=cv2.imread(img_path)
    # 获取图像大小
    height,width = pil_image.shape[:2]
    # 将坐标数据转换为float类型
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    whether_erase=False

    dis_kuan=np.linalg.norm(box_coordinates[1]-box_coordinates[0])
    dis_gao=np.linalg.norm(box_coordinates[2]-box_coordinates[1])       
    # 找到最小外接矩形
    rect = cv2.minAreaRect(box_coordinates)
    # 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect

    size_new=(size[0],size[1])
    # 生成新的最小外接矩形的四个顶点
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    new_box = np.int0(new_box)
    # 创建一个全黑的图像作为初始掩码
    mask = np.zeros((height,width), dtype=np.uint8) 
    # 在掩码上绘制新的多边形，使用白色填充
    cv2.drawContours(mask, [new_box], 0, (255), -1)
    mask = 255 - mask
    return mask,whether_erase,new_box
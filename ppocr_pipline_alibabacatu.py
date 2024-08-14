import base64
from PIL import Image
import json
import requests
# from paddleocr import PaddleOCR, draw_ocr
import os
import cv2
import numpy as np
from io import BytesIO
import numpy as np

def enlarge_box(box, delta_width, delta_height):
    # 找到最小外接矩形
    rect = cv2.minAreaRect(box)   
    # 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect
    if angle<45:
        size_new=(size[0]+delta_width,size[1]+delta_height)
    else:
        size_new=(size[0]+delta_height,size[1]+delta_width)
    # 生成新的最小外接矩形的四个顶点
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    return new_box


def Alicatu(dt_boxes,image_path):
    image_erase=cv2.imread(image_path) 
    # 获取图像大小
    img_height,img_width = image_erase.shape[:2]
    ocrCoordDTOList = []
    box_coordinates = np.array(dt_boxes, dtype=np.float32)
    box_coordinates=enlarge_box(box_coordinates,5,5)#适当扩大擦图区域提升擦图效果
    box_coordinates[:,0]=np.clip(box_coordinates[:, 0], 1, img_width-1)
    box_coordinates[:,1]=np.clip(box_coordinates[:, 1], 1, img_height-1)

     
    
    rect = cv2.minAreaRect(box_coordinates)
    # step 1: 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect
    # step 2: 计算旋转矩阵
    if angle==180:
        angle=0
    angle=angle-90
    if angle < -45:
        angle = 90 + angle
    else:
        size = (size[1], size[0])
    # 获取旋转矩阵，参数1是旋转中心点，参数2是旋转的角度，参数3是旋转的比例
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # step 3: 执行仿射变换，即旋转图片
    rotated_image = cv2.warpAffine(image_erase, rotation_matrix, (image_erase.shape[1], image_erase.shape[0]), flags=cv2.INTER_CUBIC)
    # step 4: 计算旋转后的矩形框的新坐标
    rotated_rect = cv2.transform(np.array([box_coordinates]), rotation_matrix)
    # 根据四边形的坐标，生成转换到矩形的透视变换矩阵
    M = cv2.getPerspectiveTransform(rotated_rect, box_coordinates)

    ocrCoordDTO = {
        "upLeft": {"x": int(box_coordinates[0][0]), "y": int(box_coordinates[0][1])},
        "upRight": {"x": int(box_coordinates[1][0]), "y": int(box_coordinates[1][1])},
        "downRight": {"x": int(box_coordinates[2][0]), "y": int(box_coordinates[2][1])},
        "downLeft": {"x": int(box_coordinates[3][0]), "y": int(box_coordinates[3][1])}
    }
    ocrCoordDTOList.append(ocrCoordDTO)
    #image_bytes=open(image_path, 'rb').read()
    # 将图片编码为JPEG格式的字节流（返回值是 (成功flag, 字节流)）
    is_success, buffer = cv2.imencode(".jpg", rotated_image)
    if is_success:
        # 将字节流转换为bytearray，这和open(image_path, 'rb').read() 类似
        image_bytes = np.array(buffer).tobytes()
    # 现在 image_bytes 包含了和 open(image_path, 'rb').read() 相同的二进制数据
    dataBase64=base64.b64encode(image_bytes)
    #ocrCoordDTOList=[{'downLeft': {'x': 54, 'y': 174}, 'downRight': {'x': 474, 'y': 110}, 'upLeft': {'x': 42, 'y': 92}, 'upRight': {'x': 462, 'y': 29}}, {'downLeft': {'x': 58, 'y': 256}, 'downRight': {'x': 225, 'y': 224},  'upLeft': {'x': 48, 'y': 209}, 'upRight': {'x': 215, 'y': 176}}, {'downLeft': {'x': 207, 'y': 228}, 'downRight': {'x': 475, 'y': 177}, 'upLeft': {'x': 198, 'y': 181}, 'upRight': {'x': 466, 'y': 130}}]
    ocr_info={"ocrCoordDTOList":ocrCoordDTOList}

    # python请求示例
    query_string = json.dumps({"inputs":{"src_image": dataBase64.decode('utf-8'), "text_information": json.dumps(ocr_info)}})
    work_flow = {"work_flow":[{"model_type":"text_stroke_segment","model_version":"1.0.0","model_domain":"general"}]}
    payload = {"service":"xrefine",
            "src":"text",
            "appname":"skimage",
            "src_lang":"multi-lang",
            "action_type":"custom",
            "format":"json",
            "is_compress": "false",  
            "pvid":"testzzz",
            "sent_pvid":"testyyy",
            "work_flow":json.dumps(work_flow),
            "query": query_string}
    headers={'Content-Type':'application/x-www-form-urlencoded'}
    http_url="http://11.14.177.217:20505/xr"
    r = requests.post(http_url, data=payload, headers=headers)

    data = json.loads(r.text)

    data_src = json.loads(data["result"]["response"][0]["outputJson"][0])
    imgdata = base64.b64decode(data_src["tgt_image"])

#     image_save_path="/mnt/nas/users/zhipeng.qzp/AnyText/alicatu/tmp_123."+image_format
#     with open(image_save_path, 'wb') as wf:
#         wf.write(imgdata)
#    image_xuanzhuan=cv2.imread(image_save_path)

    # 使用内存文件对象
    image_stream = BytesIO(imgdata)
    # OpenCV 不直接支持从内存中直接读取数据创建图像，因此这里先使用Pillow
    pil_image = Image.open(image_stream)
    # 将Pillow图像转换为NumPy数组，以便OpenCV可以处理
    cv_image = np.array(pil_image)
    # Pillow默认为RGB顺序，而OpenCV为BGR，所以需要转换
    image_xuanzhuan = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('/mnt/nas/users/zhipeng.qzp/AnyText/Ali_Catu/image_xuanzhuan.jpg', image_xuanzhuan)



    # 应用透视变换，将源图像上的四边形截取出来并转换到目标图像的对应四边形区域
    # 在`warpPerspective`的输出大小参数中，我们使用目标图像的宽度和高度
    warped_source = cv2.warpPerspective(image_xuanzhuan, M, (image_erase.shape[1], image_erase.shape[0]))
    # 创建一个掩码，白色区域表示我们要从warped_source传输的区域
    mask = np.zeros(image_erase.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, box_coordinates.astype(int), 255)
    # 将掩码转换为三通道
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 使用mask进行位操作，将warped_source对应区域传输到target_img上
    image_erase = cv2.bitwise_and(image_erase, cv2.bitwise_not(mask_3d))
    # cv2.imwrite('/mnt/nas/users/zhipeng.qzp/AnyText/Ali_Catu/image_erase.jpg', image_erase)
    warped_source = cv2.bitwise_and(warped_source, mask_3d)
    # cv2.imwrite('/mnt/nas/users/zhipeng.qzp/AnyText/Ali_Catu/warped_source.jpg', warped_source)
    # 合并两个图像
    final_img = cv2.add(image_erase, warped_source)
    # cv2.imwrite('/mnt/nas/users/zhipeng.qzp/AnyText/Ali_Catu/final_image.jpg', final_img )



    # inverse_rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    # back_rotated_image = cv2.warpAffine(image_xuanzhuan, inverse_rotation_matrix, (rotated_image.shape[1], rotated_image.shape[0]), flags=cv2.INTER_CUBIC)
    erased_image_path=image_path[:-4]+'_erase.png'
    # cv2.imwrite(erased_image_path, back_rotated_image)
    cv2.imwrite(erased_image_path, final_img)
    # image_path=erased_image_path
    # ocrdata = json.loads(data_src["text_information"])
    # with open("/mnt/nas/users/zhipeng.qzp/AnyText/alicatu/tmp_123.json", "w") as wf:
    #     json.dump(ocrdata, wf)


def Alicatu_evaluate(dt_boxes,image_path,save_path):
    ocrCoordDTOList = []
    box_coordinates = np.array(dt_boxes, dtype=np.float32)
    box_coordinates=enlarge_box(box_coordinates,10,10)#适当扩大擦图区域提升擦图效果
    image_erase=cv2.imread(image_path)  
    
    rect = cv2.minAreaRect(box_coordinates)
    # step 1: 计算矩形的中心、宽度、高度和旋转角
    center, size, angle = rect
    # step 2: 计算旋转矩阵
    angle=angle-90
    if angle < -45:
        angle = 90 + angle
    else:
        size = (size[1], size[0])
    # 获取旋转矩阵，参数1是旋转中心点，参数2是旋转的角度，参数3是旋转的比例
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # step 3: 执行仿射变换，即旋转图片
    rotated_image = cv2.warpAffine(image_erase, rotation_matrix, (image_erase.shape[1], image_erase.shape[0]), flags=cv2.INTER_CUBIC)
    # step 4: 计算旋转后的矩形框的新坐标
    rotated_rect = cv2.transform(np.array([box_coordinates]), rotation_matrix)
    # 根据四边形的坐标，生成转换到矩形的透视变换矩阵
    M = cv2.getPerspectiveTransform(rotated_rect, box_coordinates)

    ocrCoordDTO = {
        "upLeft": {"x": int(box_coordinates[0][0]), "y": int(box_coordinates[0][1])},
        "upRight": {"x": int(box_coordinates[1][0]), "y": int(box_coordinates[1][1])},
        "downRight": {"x": int(box_coordinates[2][0]), "y": int(box_coordinates[2][1])},
        "downLeft": {"x": int(box_coordinates[3][0]), "y": int(box_coordinates[3][1])}
    }
    ocrCoordDTOList.append(ocrCoordDTO)
    #image_bytes=open(image_path, 'rb').read()
    # 将图片编码为JPEG格式的字节流（返回值是 (成功flag, 字节流)）
    is_success, buffer = cv2.imencode(".jpg", rotated_image)
    if is_success:
        # 将字节流转换为bytearray，这和open(image_path, 'rb').read() 类似
        image_bytes = np.array(buffer).tobytes()
    # 现在 image_bytes 包含了和 open(image_path, 'rb').read() 相同的二进制数据
    dataBase64=base64.b64encode(image_bytes)
    #ocrCoordDTOList=[{'downLeft': {'x': 54, 'y': 174}, 'downRight': {'x': 474, 'y': 110}, 'upLeft': {'x': 42, 'y': 92}, 'upRight': {'x': 462, 'y': 29}}, {'downLeft': {'x': 58, 'y': 256}, 'downRight': {'x': 225, 'y': 224},  'upLeft': {'x': 48, 'y': 209}, 'upRight': {'x': 215, 'y': 176}}, {'downLeft': {'x': 207, 'y': 228}, 'downRight': {'x': 475, 'y': 177}, 'upLeft': {'x': 198, 'y': 181}, 'upRight': {'x': 466, 'y': 130}}]
    ocr_info={"ocrCoordDTOList":ocrCoordDTOList}

    # python请求示例
    query_string = json.dumps({"inputs":{"src_image": dataBase64.decode('utf-8'), "text_information": json.dumps(ocr_info)}})
    work_flow = {"work_flow":[{"model_type":"text_stroke_segment","model_version":"1.0.0","model_domain":"general"}]}
    payload = {"service":"xrefine",
            "src":"text",
            "appname":"skimage",
            "src_lang":"multi-lang",
            "action_type":"custom",
            "format":"json",
            "is_compress": "false",  
            "pvid":"testzzz",
            "sent_pvid":"testyyy",
            "work_flow":json.dumps(work_flow),
            "query": query_string}
    headers={'Content-Type':'application/x-www-form-urlencoded'}
    http_url="http://11.14.177.217:20505/xr"
    r = requests.post(http_url, data=payload, headers=headers)

    data = json.loads(r.text)

    data_src = json.loads(data["result"]["response"][0]["outputJson"][0])
    imgdata = base64.b64decode(data_src["tgt_image"])

    # 使用内存文件对象
    image_stream = BytesIO(imgdata)
    # OpenCV 不直接支持从内存中直接读取数据创建图像，因此这里先使用Pillow
    pil_image = Image.open(image_stream)
    # 将Pillow图像转换为NumPy数组，以便OpenCV可以处理
    cv_image = np.array(pil_image)
    # Pillow默认为RGB顺序，而OpenCV为BGR，所以需要转换
    image_xuanzhuan = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)


    # 应用透视变换，将源图像上的四边形截取出来并转换到目标图像的对应四边形区域
    # 在`warpPerspective`的输出大小参数中，我们使用目标图像的宽度和高度
    warped_source = cv2.warpPerspective(image_xuanzhuan, M, (image_erase.shape[1], image_erase.shape[0]))
    # 创建一个掩码，白色区域表示我们要从warped_source传输的区域
    mask = np.zeros(image_erase.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, box_coordinates.astype(int), 255)
    # 将掩码转换为三通道
    mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 使用mask进行位操作，将warped_source对应区域传输到target_img上
    image_erase = cv2.bitwise_and(image_erase, cv2.bitwise_not(mask_3d))
    warped_source = cv2.bitwise_and(warped_source, mask_3d)
    # 合并两个图像
    final_img = cv2.add(image_erase, warped_source)

    image_name=(img_path.split('/'))[-1][:-4]
    # erased_image_path=image_path[:-4]+'_erase.png'
    erased_image_path=os.path.join(save_path,f'erase_{image_name}.png')
    # cv2.imwrite(erased_image_path, back_rotated_image)
    cv2.imwrite(erased_image_path, final_img)
    return erased_image_path







if __name__=='__main__':
    image_path='/mnt/nas/users/zhipeng.qzp/AnyText/alicatu/001253506.jpg'
    ocr = PaddleOCR(lang="en") # 首次执行会自动下载模型文件
    result = ocr.ocr(image_path)
    result=result[0]
    dt_boxes = [line[0] for line in result]
    Alicatu(dt_boxes,image_path)

    # image = Image.open(image_path).convert('RGB')
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, dt_boxes, txts, scores, font_path='/mnt/nas/users/zhipeng.qzp/AnyText/PaddleOCR/doc/fonts/simfang.ttf')
    # im_show = Image.fromarray(im_show)
    # filename = os.path.basename(image_path)
    # save_dir='/mnt/nas/users/zhipeng.qzp/AnyText/alicatu/'
    # save_path=save_dir+filename
    # im_show.save(save_path)



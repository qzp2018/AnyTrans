from PIL import Image, ImageDraw
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import cv2
from util import resize_image,resize_mask,enlarge_box_bigger,image_crop,resize_mask_returnbox_suokuan,resize_image_boxes
from http import HTTPStatus
import dashscope
import os
from PIL import Image
import io
import json
import re
import requests
import sys
import random
import concurrent.futures
import time
#sys.path.append("/mnt/nas/users/zhipeng.qzp/AnyText/PaddleOCR") # 将module1所在的文件夹路径放入sys.path中
# from PaddleOCR.tools.infer.predict_system_anytext import TextSystem
# import PaddleOCR.tools.infer.utility as utility


from dashscope import MultiModalConversation

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from util import save_images
import argparse

from Ali_Catu.ppocr_pipline_alibabacatu import Alicatu



params = {
    "show_debug": True,
    "image_count": 1,
    "ddim_steps": 20,
}
pipe = pipeline('my-anytext-task', model='damo/cv_anytext_text_generation_editing', model_revision='v1.1.2')

# 加载模型
def call_with_prompt_llm_all(input_text):
    dashscope.api_key =''
    output_text = ''.join([f'<box{i+1}>{text}</box{i+1}>' for i, text in enumerate(input_text)])
    prompt_template='Translate the following phrase from Chinese into English.\nChinese: <box1>白虎</box1>\nEnglish: <box1>white tiger</box1>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>生日</box1><box2>快乐</box2>\nEnglish: <box1>Happy</box1><box2>Birthday</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>新年</box1><box2>快乐</box2>\nEnglish: <box1>Happy</box1><box2>New Year</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>@交通北京</box1><box2>神驼2号</box2><box3>北京养护集团</box3>\nEnglish: <box1>@Traffic Beijing</box1><box2>Giant Camel 2</box2><box3>Beijng Maintenance Group</box3>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>青草依依</box1><box2>请您爱惜</box2>\nEnglish: <box1>Grass is green and fresh</box1><box2>please cherish it</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>门</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>严禁烟火</box5><box6>比我高</box6><box7>买票</box7>\nEnglish: <box1>Gate</box1><box2>Stamp Office</box2><box3>Ticket Counter</box3><box4>No Dogs Allowed</box4><box5>No Fireworks</box5><box6>Higher than me</box6><box7>Buy tickets</box7>\n\nTranslate the following sentence from Chinese into English, keep the length similar and no more than 20 letters.\n'+output_text+'\nEnglish '
    response = dashscope.Generation.call(
        'qwen2-7b-instruct',
        prompt=prompt_template,
        result_format='string')
    wrong_count=0
    while response.status_code != HTTPStatus.OK and wrong_count<=8:
        time.sleep(8)  # 在请求失败后等待8秒钟再进行重试
        wrong_count+=1    
    if response.status_code == HTTPStatus.OK:
        result_txt=response.output['text']
        box_pattern = re.compile(r'<box\d+>(.*?)</box\d+>')
        matches = box_pattern.findall(result_txt)
        if len(input_text)==len(matches):
            result_txt=matches
            return True,result_txt
        elif len(input_text)==1:#处理单个box的情况
            return True,[result_txt]
        else:
            return False,result_txt
    else:
        time.sleep(5)  # 在请求失败后等待2秒钟再进行重试
        result_txt='failed'
        return False,result_txt


# 加载模型
def call_with_prompt_llm_all_boxbybox(input_text):
    dashscope.api_key =''
    output_text='<box1>'+input_text+'</box1>'
    prompt_template='Translate the following phrase from Chinese into English.\nChinese: <box1>白虎</box1>\nEnglish: <box1>white tiger</box1>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>生日</box1><box2>快乐</box2>\nEnglish: <box1>Happy</box1><box2>Birthday</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>新年</box1><box2>快乐</box2>\nEnglish: <box1>Happy</box1><box2>New Year</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>@交通北京</box1><box2>神驼2号</box2><box3>北京养护集团</box3>\nEnglish: <box1>@Traffic Beijing</box1><box2>Giant Camel 2</box2><box3>Beijng Maintenance Group</box3>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>青草依依</box1><box2>请您爱惜</box2>\nEnglish: <box1>Grass is green and fresh</box1><box2>please cherish it</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>门</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>严禁烟火</box5><box6>比我高</box6><box7>买票</box7>\nEnglish: <box1>Gate</box1><box2>Stamp Office</box2><box3>Ticket Counter</box3><box4>No Dogs Allowed</box4><box5>No Fireworks</box5><box6>Higher than me</box6><box7>Buy tickets</box7>\n\nTranslate the following sentence from Chinese into English, keep the length similar and no more than 20 letters.\n'+output_text+'\nEnglish '
    # prompt_template='Translate the following phrase from Chinese into Japanese.\nChinese: <box1>狮子</box1>\nJapanese: <box1>ライオン</box1>\n\nTranslate the following phase from Chinese into Japanese.\nChinese: <box1>生日</box1><box2>快乐</box2>\nJapanese: <box1>誕生日</box1><box2>しあわせ</box2>\n\nTranslate the following phase from Chinese into Japanese.\nChinese: <box1>新年</box1><box2>快乐</box2>\nJapanese: <box1>あけまし</box1><box2>ておめでとう</box2>\n\nTranslate the following phase from Chinese into Japanese.\nChinese: <box1>@交通北京</box1><box2>神驼</box2><box3>北京养护集团</box3>\nJapanese: <box1>@トラフィック北京</box1><box2>神ラクダ</box2><box3>北京メンテナンスグループ</box3>\n\nTranslate the following phase from Chinese into Japanese.\nChinese: <box1>青草依依</box1><box2>请您爱惜</box2>\nJapanese: <box1>緑の草</box1><box2>大切にしてください</box2>\n\nTranslate the following phase from Chinese into Japanese.\nChinese: <box1>门</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>禁止抽烟</box5><box6>比我高</box6><box7>买票</box7>\nJapanese: <box1>ドア</box1><box2>オークオフィス</box2><box3>改札口</box3><box4>犬お断り</box4><box5>禁煙</box5><box6>私より背が高い</box6><box7>切符を買う</box7>\n\nTranslate the following phase from Chinese into Japanese, keep the length similar and no more than 20 letters.\n'+output_text+'\nJapanese '
    #prompt_template='Translate the following phrase from Chinese into Korean.\nChinese: <box1>白虎</box1>\nKorean: <box1>백호</box1>\n\nTranslate the following phase from Chinese into Korean.\nChinese: <box1>生日</box1><box2>快乐</box2>\nKorean: <box1>생일</box1><box2>축하해요</box2>\n\nTranslate the following phase from Chinese into Korean.\nChinese: <box1>新年</box1><box2>快乐</box2>\nKorean: <box1>새해</box1><box2>복 많이 받으세요</box2>\n\nTranslate the following phase from Chinese into Korean.\nChinese: <box1>@交通北京</box1><box2>神驼</box2><box3>北京养护集团</box3>\nKorean: <box1>@교통 베이징</box1><box2>신낙타</box2><box3>베이징 유지 관리 그룹</box3>\n\nTranslate the following phase from Chinese into Korean.\nChinese: <box1>青草依依</box1><box2>请您爱惜</box2>\nKorean: <box1>녹색 풀</box1><box2>소중히 여겨주세요</box2>\n\nTranslate the following phase from Chinese into Korean.\nChinese: <box1>门</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>严禁烟火</box5><box6>比我高</box6><box7>买票</box7>\nKorean: <box1>문</box1><box2>오크오피스</box2><box3>개찰구</box3><box4>개는 허용되지 않습니다</box4><box5>금연</box5><box6>나보다 키가 크다</box6><box7>티켓 구매</box7>\n\nTranslate the following phase from Chinese into Korean, keep the length similar and no more than 20 letters.\n'+output_text+'\nKorean '
    #prompt_template='Translate the following sentence from English into Chinese.\nEnglish: <box1>white tiger</box1>\nChinese: <box1>白虎</box1>\n\nTranslate the following sentence from English into Chinese.\nEnglish:  <box1>Happy</box1><box2>Birthday</box2>\nChinese: <box1>生日</box1><box2>快乐</box2>\n\nTranslate the following sentence from English into Chinese.\nEnglish: <box1>Happy</box1><box2>New Year</box2>\nChinese: <box1>新年</box1><box2>快乐</box2>\n\nTranslate the following sentence from English into Chinese.\nEnglish: <box1>@Traffic Beijing</box1><box2>Giant Camel 2</box2><box3>Beijng Maintenance Group</box3>\nChinese: <box1>@交通北京</box1><box2>神驼2号</box2><box3>北京养护集团</box3>\n\nTranslate the following sentence from English into Chinese.\nEnglish: <box1>Grass is green and fresh</box1><box2>please be careful of it</box2>\nChinese: <box1>青草依依</box1><box2>请您爱惜</box2>\n\nTranslate the following sentence from English into Chinese.\nEnglish: <box1>gate</box1><box2>Stamp Office</box2><box3>Ticket Counter</box3><box4>No Dogs Allowed</box4><box5>No Fireworks</box5><box6>taller than me</box6><box7>Buy tickets</box7>\nChinese: <box1>门</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>严禁烟火</box5><box6>比我高</box6><box7>买票</box7>\n\n'+'Translate the following sentence from English into Chinese.\nEnglish: '+output_text+'\nChinese: '
    #prompt_template='Translate the following sentence from Japanese into Chinese.\nJapanese: <box1>ホワイトタイガー</box1>\nChinese: <box1>白虎</box1>\n\nTranslate the following sentence from Japanese into Chinese.\nJapanese: <box1>博多ラーメン</box1>\nChinese: <box1>博多拉面</box1>\n\nTranslate the following sentence from Japanese into Chinese.\nJapanese: <box1>おすすめワイン</box1><box2>500円</box2>\nChinese: <box1>推荐的酒</box1><box2>500日元</box2>\n\nTranslate the following sentence from Japanese into Chinese.\nJapanese: <box1>@トラフィック北京</box1><box2>神陀2号</box2><box3>北京メンテナンスグループ</box3>\nChinese: <box1>@交通北京</box1><box2>神驼2号</box2><box3>北京养护集团</box3>\n\nTranslate the following sentence from Japanese into Chinese.\nJapanese: <box1>緑の草</box1><box2>大切にしてください</box2>\nChinese:<box1>青草依依</box1><box2>请您爱惜</box2>\n\nTranslate the following sentence from Japanese into Chinese.\nJapanese: <box1>ドア</box1><box2>オークオフィス</box2><box3>改札口</box3><box4>犬お断り</box4><box5>禁煙</box5><box6>私より背が高い</box6><box7>チケットを買う</box7>\nChinese: <box1>验票处</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>严禁烟火</box5><box6>比我高</box6><box7>买票喽</box7>\n\nTranslate the following sentence from Japanese into Chinese.\nJapanese: '+output_text+'\nChinese: '
    #prompt_template='Translate the following sentence from Korean into Chinese.\nKorean: <box1>백호</box1>\nChinese: <box1>白虎</box1>\n\nTranslate the following sentence from Korean into Chinese.\nKorean: <box1>손으로 뽑은 국수</box1>\nChinese: <box1>拉面</box1>\n\nTranslate the following sentence from Korean into Chinese.\nKorean: <box1>추천 와인</box1><box2>500위안</box2>\nChinese: <box1>推荐的酒</box1><box2>500元</box2>\n\nTranslate the following sentence from Korean into Chinese.\nKorean: <box1>@ 교통 베이징</box1><box2>센투오 2호</box2><box3>베이징 유지 관리 그룹</box3>\nChinese: <box1>@交通北京</box1><box2>神驼2号</box2><box3>北京养护集团</box3>\n\nTranslate the following sentence from Korean into Chinese.\nKorean: <box1>녹색 풀</box1><box2>소중히 여겨주세요</box2>\nChinese:<box1>青草依依</box1><box2>请您爱惜</box2>\n\nTranslate the following sentence from Korean into Chinese.\nKorean: <box1>매표소</box1><box2>오크오피스</box2><box3>오크오피스</box3><box4>개는 허용되지 않습니다</box4><box5>금연</box5><box6>나보다 키가 크다</box6><box7>티켓을 사다</box7>\nChinese: <box1>验票处</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>严禁烟火</box5><box6>比我高</box6><box7>买票喽</box7>\n\nTranslate the following sentence from Korean into Chinese.\nKorean: '+output_text+'\nChinese: '
    response = dashscope.Generation.call(
        'qwen2-7b-instruct',
        prompt=prompt_template,
        result_format='string')
    wrong_count=0
    while response.status_code != HTTPStatus.OK and wrong_count<=8:
        time.sleep(8)  # 在请求失败后等待8秒钟再进行重试
        wrong_count+=1    
    if response.status_code == HTTPStatus.OK:
        result_txt=response.output['text']
        box_pattern = re.compile(r'<box\d+>(.*?)</box\d+>')
        matches = box_pattern.findall(result_txt)
        if len(matches)==1:
            result_txt=matches[0]
            return result_txt
        else:
            return result_txt
    else:
        time.sleep(5)  # 在请求失败后等待2秒钟再进行重试
        result_txt='failed'
        return result_txt
  



def translaor_using_piple(input_txt):
    # trans_txt=translator(input_txt)[0]['translation_text']
    responses_all=[]
    for tmp_txt in input_txt:
        tmp_response=call_with_prompt_llm_all_boxbybox(tmp_txt)
        responses_all.append(tmp_response)
    return True,responses_all 



def create_mask(pil_image, box_coordinates):
    # 获取图像大小
    height,width  = pil_image.shape[:2]
    image_size = (height,width )
    # 创建一个全黑的图像作为初始掩码
    mask = np.zeros(image_size, dtype=np.uint8)
    # 将坐标数据转换为整数类型
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    # 在掩码上绘制多边形，使用白色填充（即在黑色掩码上绘制黑色区域）
    cv2.fillPoly(mask, [box_coordinates], color=(255, 255, 255))
    mask=255-mask
    return mask

# ocr = PaddleOCR(use_angle_cls=True, lang="ch",ocr_version='PP-OCRv4')


def PPOCR_pipline(img_path):
    # text_sys = TextSystem(args)
    pil_image = cv2.imread(img_path)
    result=ocr.ocr(img_path)
    result=result[0]    
    dt_boxes = [line[0] for line in result]

    pil_image,new_dt_boxes= resize_image_boxes(pil_image,dt_boxes, max_length=768)
    resize_image_path=img_path[:-4]+'_resize.jpg'
    cv2.imwrite(resize_image_path,pil_image )
    img_path=resize_image_path

    # if len(dt_boxes)!=max_value:
    #     print(f'{img_path} not corresponding!')
    #     return None    
    for i,tmp_box in enumerate(new_dt_boxes):
        new_dt_boxes[i]=enlarge_box_bigger(tmp_box)
    all_txts = [line[1][0] for line in result]
    all_text_ocr=all_txts
    if all_text_ocr==[]:
        return None
    judge,translate_responses=call_with_prompt_llm_all(all_text_ocr)
    if judge==False:
        judge,translate_responses=translaor_using_piple(all_text_ocr)
    #judge,translate_responses=call_with_prompt_ours_llm(all_text_ocr)
    if judge==False:#如何LLM翻译格式出错，就不进行图像翻译
        translation_log=img_path[:-4]+"_wrongtranslation_log.txt"
        log_file = open(translation_log, "w")
        log_file.write(str(all_text_ocr) + "\t")
        log_file.write(str(translate_responses) + "\t")        
        return None
    trans_mode='ch2en'
    image = np.array(pil_image)
    # 转换图像通道顺序为 BGR
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.clip(1, 255) # 截断像素 
    ori_image_path=img_path

    translation_log=img_path[:-4]+"translation_log.txt"
    log_file = open(translation_log, "w")
    evaluate_log=img_path[:-4]+"evaluate_log.txt"
    evaluate_log_file = open(evaluate_log, "w")

    for idx in range(len(new_dt_boxes)):#针对多个部分逐个翻译
        boxes=new_dt_boxes[idx]
        mask=create_mask(pil_image,boxes)
        word_count_old=len(all_text_ocr[idx])
        tmp_trans=all_txts[idx]
        trans_mode='ch2en'
        untranslate=False
        tmp_pattern = re.compile('[a-zA-Z0-9]{1,}')
        if re.search(tmp_pattern, tmp_trans):
            untranslate=True
            trans_mode='others'
        # word_count_new=len(translate_responses[idx])
        try :#防止word_count_new=len(translate_responses[idx])报错
            word_count_new=len(translate_responses[idx])
        except:
            return None
        resized_mask,whether_erase,return_box=resize_mask_returnbox_suokuan(ori_image_path,boxes,word_count_old,word_count_new,trans_mode)

        # if idx>=1:
        #     mask = resize_image(mask, max_length=768) 
        #     resized_mask=resize_image(resized_mask, max_length=768)
        #     image=resize_image(image, max_length=768)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        resized_masked_image=cv2.bitwise_and(image, image, mask=resized_mask)
        if idx==0:#一开始的时候mask的图片大小需要修正
            masked_image = resize_image(masked_image, max_length=768) 
            resized_masked_image= resize_image(resized_masked_image, max_length=768)
            # original_image=resize_image(resized_masked_image, max_length=768)
            
        # 将 NumPy 数组保存为图像文件
        masked_image_path=img_path[:-4]+'_masked_'+str(idx)+'.png'
        resized_masked_image_path=img_path[:-4]+'_resized_masked_'+str(idx)+'.png'
        cv2.imwrite(masked_image_path, masked_image)
        cv2.imwrite(resized_masked_image_path, resized_masked_image)
        #判断是否需要擦图
        if whether_erase==True and untranslate==False:
            Alicatu(boxes,ori_image_path)
            erased_image_path=ori_image_path[:-4]+'_erase.png'
            ori_image_path=erased_image_path#需要编辑的图片路径更新
        #翻译
        txts=all_txts[idx] #OCR识别原文结果
        print(txts)
        # 将识别结果写入日志文件
        log_file.write(txts + "\t")
        #response=call_with_local_file(txts,ori_image_path2)#qwen-vl
        try:
            response=translate_responses[idx]#qwen
        except:#针对llm翻译格式不确定，导致翻译结果变少的情况，使用机器翻译
            response=translaor_using_piple(txts)
        # 将翻译结果写入日志文件
        try:
            log_file.write(response + "\n")
        except:
            return None
        if response==txts:#如果是专有名词无需翻译，就不进行后续图片编辑，保存现有的图像
            inpainted_image_path=img_path[:-4]+'_inpainted.png'
            cv2.imwrite(inpainted_image_path, image)
            inpainted_image=image#保持图片不修改

            evaluate_log_file.write(str(txts) + "\t")
            evaluate_log_file.write(response + "\t")
            evaluate_log_file.write(response+'\n')
        elif untranslate==True:
            inpainted_image_path=img_path[:-4]+'_inpainted.png'
            cv2.imwrite(inpainted_image_path, image)
            inpainted_image=image#保持图片不修改

            evaluate_log_file.write(str(txts) + "\t")
            evaluate_log_file.write(str(tmp_trans) + "\t")
            evaluate_log_file.write(str(tmp_trans)+'\n')
        else:
            # 3. text editing
            mode = 'text-editing'
            tmp_prompt='一张海报，上面写着' +'"'+str(response.strip('"'))+'"'
            input_data = {
                "prompt": tmp_prompt,
                "seed": 94081527,
                "draw_pos":resized_masked_image_path,
                "ori_image":ori_image_path,
            }

            try:
                results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
            except:
                translation_log=img_path[:-4]+"number_wrongtranslation_log.txt"
                log_file = open(translation_log, "w")
                log_file.write(str(all_text_ocr) + "\t")
                log_file.write(str(translate_responses) + "\t") 
                return None 
            if rtn_code >= 0:
                inpainted_image=results[0]
                inpainted_image= cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR)
                inpainted_image_path=img_path[:-4]+'_inpainted.png'
                cv2.imwrite(inpainted_image_path, inpainted_image)
                crop_image_path=image_crop(inpainted_image,return_box,inpainted_image_path,idx)
                crop_result=evaluate_ocr.ocr(crop_image_path)
                if crop_result[0]==None:
                    crop_text_write='failed'
                else:
                    crop_text= [line[1][0] for line in crop_result[0]]
                    crop_text_write=' '.join(crop_text)

                #如果anytext的编辑结果不佳，则重新编辑
                Anytext_editing_count=0
                while response!=crop_text_write and Anytext_editing_count<5:
                    Anytext_editing_count+=1
                    random_number = random.randint(1, 10000000)
                    mode = 'text-editing'
                    tmp_prompt='一张海报，上面写着' +'"'+str(response.strip('"'))+'"'
                    input_data = {
                        "prompt": tmp_prompt,
                        "seed": random_number,
                        "draw_pos":resized_masked_image_path,
                        "ori_image":ori_image_path,
                    }
                    results, rtn_code, rtn_warning, debug_info = pipe(input_data, mode=mode, **params)
                    inpainted_image=results[0]
                    inpainted_image= cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR)
                    inpainted_image_path_reedit=img_path[:-4]+'_inpainted_reedit.png'
                    cv2.imwrite(inpainted_image_path_reedit, inpainted_image)
                    crop_image_path=image_crop(inpainted_image,return_box,inpainted_image_path_reedit,idx)
                    #crop_result=baidu_ocr(crop_image_path,'CHN_ENG')
                    crop_result=evaluate_ocr.ocr(crop_image_path)
                    if crop_result[0]==None:
                        crop_text_write='failed'
                    else:
                        crop_text= [line[1][0] for line in crop_result[0]]
                        crop_text_write=' '.join(crop_text)
                    if Anytext_editing_count==5 or response==crop_text_write:
                        inpainted_image_path=img_path[:-4]+'_inpainted.png'
                        cv2.imwrite(inpainted_image_path, inpainted_image)              


                evaluate_log_file.write(str(txts) + "\t")
                evaluate_log_file.write(response + "\t")
                evaluate_log_file.write(crop_text_write+'\n')
                

            else:#如果没有成功生成翻译且编辑后的图片，就使用原图
                inpainted_image_path=img_path[:-4]+'_inpainted.png'
                cv2.imwrite(inpainted_image_path, image)
                inpainted_image=image#保持图片不修改
        ori_image_path=inpainted_image_path #  编辑图片更新
        image=inpainted_image.clip(1, 255) # 更新后续需要mask的位置 同时截断像素 



if __name__ == '__main__':
    folder_path='/mnt/nas/users/zhipeng.qzp/AnyText/Duoyuyan_New/CH2EN_hard/New_Chinese_qwen2_7B/'
    image_path_list=[]
    ocr = PaddleOCR(lang="ch") # 首次执行会自动下载模型文件
    evaluate_ocr = PaddleOCR(lang="en") # 首次执行会自动下载模型文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是图片文件
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # 构建图片的完整路径
            image_path = os.path.join(folder_path, filename)
            image_path_list.append(image_path)
    # 调用图像处理函数进行处理
    for i in range(250):
        path=os.path.join(folder_path,f'ch_{i+1}.jpg')
        if path in image_path_list:
            print(f"Processed image: {path}")
            # result = ocr.ocr(path)
            PPOCR_pipline(path)
            
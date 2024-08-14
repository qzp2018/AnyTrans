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
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline as trans_pipeline
import random
import concurrent.futures
import time

from dashscope import MultiModalConversation

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from util import save_images
import argparse

from catu import Alicatu



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
        'qwen1.5-7b-chat',
        prompt=prompt_template,
        result_format='string')
    wrong_count=0
    while response.status_code != HTTPStatus.OK and wrong_count<=8:
        time.sleep(8)  
        wrong_count+=1    
    if response.status_code == HTTPStatus.OK:
        result_txt=response.output['text']
        box_pattern = re.compile(r'<box\d+>(.*?)</box\d+>')
        matches = box_pattern.findall(result_txt)
        if len(input_text)==len(matches):
            result_txt=matches
            return True,result_txt
        elif len(input_text)==1:
            return True,[result_txt]
        else:
            return False,result_txt
    else:
        time.sleep(5)  
        result_txt='failed'
        return False,result_txt


# 加载模型
def call_with_prompt_llm_all_boxbybox(input_text):
    dashscope.api_key =''
    output_text='<box1>'+input_text+'</box1>'
    prompt_template='Translate the following phrase from Chinese into English.\nChinese: <box1>白虎</box1>\nEnglish: <box1>white tiger</box1>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>生日</box1><box2>快乐</box2>\nEnglish: <box1>Happy</box1><box2>Birthday</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>新年</box1><box2>快乐</box2>\nEnglish: <box1>Happy</box1><box2>New Year</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>@交通北京</box1><box2>神驼2号</box2><box3>北京养护集团</box3>\nEnglish: <box1>@Traffic Beijing</box1><box2>Giant Camel 2</box2><box3>Beijng Maintenance Group</box3>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>青草依依</box1><box2>请您爱惜</box2>\nEnglish: <box1>Grass is green and fresh</box1><box2>please cherish it</box2>\n\nTranslate the following sentence from Chinese into English.\nChinese: <box1>门</box1><box2>橡札所</box2><box3>票口</box3><box4>禁止携犬入内</box4><box5>严禁烟火</box5><box6>比我高</box6><box7>买票</box7>\nEnglish: <box1>Gate</box1><box2>Stamp Office</box2><box3>Ticket Counter</box3><box4>No Dogs Allowed</box4><box5>No Fireworks</box5><box6>Higher than me</box6><box7>Buy tickets</box7>\n\nTranslate the following sentence from Chinese into English, keep the length similar and no more than 20 letters.\n'+output_text+'\nEnglish '
    response = dashscope.Generation.call(
        'qwen1.5-7b-chat',
        prompt=prompt_template,
        result_format='string')
    wrong_count=0
    while response.status_code != HTTPStatus.OK and wrong_count<=8:
        time.sleep(8)  
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
        time.sleep(5)  
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
    height,width  = pil_image.shape[:2]
    image_size = (height,width )
    mask = np.zeros(image_size, dtype=np.uint8)
    box_coordinates = np.array(box_coordinates, dtype=np.int32)
    cv2.fillPoly(mask, [box_coordinates], color=(255, 255, 255))
    mask=255-mask
    return mask


def PPOCR_pipline(img_path):
    pil_image = cv2.imread(img_path)
    result=ocr.ocr(img_path)
    result=result[0]    
    dt_boxes = [line[0] for line in result]

    pil_image,new_dt_boxes= resize_image_boxes(pil_image,dt_boxes, max_length=768)
    resize_image_path=img_path[:-4]+'_resize.jpg'
    cv2.imwrite(resize_image_path,pil_image )
    img_path=resize_image_path
    for i,tmp_box in enumerate(new_dt_boxes):
        new_dt_boxes[i]=enlarge_box_bigger(tmp_box)
    all_txts = [line[1][0] for line in result]
    all_text_ocr=all_txts
    if all_text_ocr==[]:
        return None
    judge,translate_responses=call_with_prompt_llm_all(all_text_ocr)
    if judge==False:
        judge,translate_responses=translaor_using_piple(all_text_ocr)
    if judge==False:
        translation_log=img_path[:-4]+"_wrongtranslation_log.txt"
        log_file = open(translation_log, "w")
        log_file.write(str(all_text_ocr) + "\t")
        log_file.write(str(translate_responses) + "\t")        
        return None
    trans_mode='ch2en'
    image = np.array(pil_image)
    image = image.clip(1, 255) 
    ori_image_path=img_path

    translation_log=img_path[:-4]+"translation_log.txt"
    log_file = open(translation_log, "w")
    evaluate_log=img_path[:-4]+"evaluate_log.txt"
    evaluate_log_file = open(evaluate_log, "w")

    for idx in range(len(new_dt_boxes)):
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
        try :
            word_count_new=len(translate_responses[idx])
        except:
            return None
        resized_mask,whether_erase,return_box=resize_mask_returnbox_suokuan(ori_image_path,boxes,word_count_old,word_count_new,trans_mode)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        resized_masked_image=cv2.bitwise_and(image, image, mask=resized_mask)
        if idx==0:
            masked_image = resize_image(masked_image, max_length=768) 
            resized_masked_image= resize_image(resized_masked_image, max_length=768)

        masked_image_path=img_path[:-4]+'_masked_'+str(idx)+'.png'
        resized_masked_image_path=img_path[:-4]+'_resized_masked_'+str(idx)+'.png'
        cv2.imwrite(masked_image_path, masked_image)
        cv2.imwrite(resized_masked_image_path, resized_masked_image)
        if whether_erase==True and untranslate==False:
            Alicatu(boxes,ori_image_path)
            erased_image_path=ori_image_path[:-4]+'_erase.png'
            ori_image_path=erased_image_path
        txts=all_txts[idx] 
        print(txts)
        log_file.write(txts + "\t")
        try:
            response=translate_responses[idx]#
        except:
            response=translaor_using_piple(txts)
        try:
            log_file.write(response + "\n")
        except:
            return None
        if response==txts:
            inpainted_image_path=img_path[:-4]+'_inpainted.png'
            cv2.imwrite(inpainted_image_path, image)
            inpainted_image=image

            evaluate_log_file.write(str(txts) + "\t")
            evaluate_log_file.write(response + "\t")
            evaluate_log_file.write(response+'\n')
        elif untranslate==True:
            inpainted_image_path=img_path[:-4]+'_inpainted.png'
            cv2.imwrite(inpainted_image_path, image)
            inpainted_image=image

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
                

            else:#
                inpainted_image_path=img_path[:-4]+'_inpainted.png'
                cv2.imwrite(inpainted_image_path, image)
                inpainted_image=image#
        ori_image_path=inpainted_image_path #  
        image=inpainted_image.clip(1, 255) # 



if __name__ == '__main__':
    folder_path=''
    image_path_list=[]
    ocr = PaddleOCR(lang="ch") # 
    evaluate_ocr = PaddleOCR(lang="en") #
    for filename in os.listdir(folder_path):
      
        if filename.endswith('.png') or filename.endswith('.jpg'):
          
            image_path = os.path.join(folder_path, filename)
            image_path_list.append(image_path)
    for i in range(250):
        path=os.path.join(folder_path,f'ch_{i+1}.jpg')
        if path in image_path_list:
            print(f"Processed image: {path}")

            PPOCR_pipline(path)
            
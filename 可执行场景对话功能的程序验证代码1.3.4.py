# %%
# -*- coding: utf-8 -*
import gradio as gr
import os
import torch
from PIL import Image
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
from lavis.models import load_model_and_preprocess
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

import librosa
import soundfile as sf
import requests
import shutil
import json
#pip install Baidu-Aip
import numpy as np
import scipy.io.wavfile as wav
import datetime
import io

# %%
dict_  = []
#Chatbot 的响应函数

def chatgpt_options(content):#这个函数调用chatgpt，输入文本输出文本
    url = "https://openai.api2d.net/v1/chat/completions"

    headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer fkxxxxx' # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
    }
    data = {
       "model": "gpt-3.5-turbo",
       "messages": [
       {"role": "system", "content": "You are a helpful assistant that helps people to complete daily conversations."},
       {"role": "user", "content": content}
       ]
    }
    response = requests.post(url, headers=headers, json=data)
    # 检查API响应的状态码
    json_response = response.json()
    choices = json_response['choices']

    if choices:
        choice = choices[0]
        message = choice.get('message')
        if message:
            content = message.get('content')
            if content:
                callback = content
                return callback
            else:
                return '1.Content not found in message.'
        else:
            return '1.Message not found in choice.'
    else:
        return '1.No choices found in response.'
    

def imgcaption(image_file):#这个函数输入图片输出文本
    # 将图片文件从临时位置复制到指定路径，以保存到本地

    #output_path = "/imgcap/image.png"  # 替换为您希望保存图片的本地路径
    #shutil.copyfile(image_file.name, output_path)
    #with open(image_file,"rb") as f:
       # imagepath = f
    imagepath = image_file#似乎出错的地方就在这里，imagepath被赋的应该是一个“/image/name.jpg”形式的路径，在组件imagepath = gr.Image()里似乎不能直接获得图片的地址。这个组件可以通过type="filepath","pil"来输出str形式的路径或者pil图像
    raw_image = Image.open(imagepath).convert("RGB")#这行代码打开指定路径下的图片文件，并转换为RGB格式的图像
    #准备问题
    question = "What kind of communication scene does this belongs to?"
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)
    answer = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")#这段代码是调用lavis库的blip模型根据图像进行问答。我希望获得的答案是图像属于哪种沟通场景。
    print(answer)
    return answer

def attipredict():
    return 5
# 定义处理函数
def attitudeoptions(option,history):#这个函数通过输入对应键的value，来选择对应的值作为输出的文本
        print(option)
        prompt_dict = {
            0:"我完全不同意你的观点。",
            1:"我对你的观点持有很大的保留。",
            2:"我对你的观点持有很多保留，无法同意。",
            3:"我对你的观点持有较多的保留，不能完全同意。",
            4:"我对你的观点持有一些保留，不完全同意。",
            5:"我可以理解你的观点，但我还有一些疑虑。",
            6:"我在某些方面同意你的看法，但在其他方面我有不同的看法。",
            7:"我对你的观点有些保留，但在某些方面可以认可。",
            8:"我大部分同意你的观点，但还有一些考虑因素。",
            9:"我很赞同你的观点，但还有一些细微差异。",
            10:"毫无疑问，我完全同意！"}
        key = option
        print(key)
        prompt_text = prompt_dict[key]
        print(prompt_text)
        history.append([prompt_text, None])
        return history

def eventdia(image, history, Dict):
        # 在这里编写处理图片和历史数据的代码
        scene = imgcaption(image)  # 这里通过imgcaption函数获得图像对应的场景“scene”
        if history == []:#说明是新对话。

            content = "假如我身处" + str(scene) + "，我有可能说些什么？请列举五句我可能说的话。"
            dict_content = chatgpt_options(content)#这里将上一行组合的问题输送给chatgpt生成回答
            print(dict_content)
            Dict_content = dict_content
            Dict.clear()#Dict是我设置的一个会话状态用来储存可供选择的文本。在这里我将它清空用来存储新生成的回答
            for line in dict_content.split('\n'):#这段for循环会将str形式的回答转化成固定格式的字典
              parts = line.split('. ')
              Dict[parts[0]] = parts[1]
            global dict_
            dict_=  Dict
            Content = content
            return [Dict_content,Content]
        if history != []:#说明对话已经开始
            last_item = history[-2]#这里要获得的是聊天记录里的倒数第二句话，也就是需要回复的内容
            callback_content = last_item[-1]
            print(callback_content)
            atti_list = history[-1]
            prompt_text = atti_list[0]#这里要获得的是本次回复里用户想表达的态度。
            print(prompt_text)
            content = "我在"+str(scene)+"发生了对话，对方回复:" + str(callback_content) + "，我的态度是" + str(prompt_text) + "请给我5句补足态度背后论据的回复。"
            print(content)
            dict_content = chatgpt_options(content)#同上
            Dict_content = dict_content
            Dict.clear()
            for line in dict_content.split('\n'):
              parts = line.split('. ')
              Dict[parts[0]] = parts[1]
            Content = content

            return [Dict_content,Content]
    
options = ["1", "2", "3", "4", "5","刷新"]
default_options = "Option 1"

def optionmsg(option,Dict, Content,history):#这个函数用来选择要发送的回复内容
        dict_content = []
        while True:#这段while循环用来确定输入的值是否在范围里，唯一一个不在范围里的选项“刷新”将触发else事件
            if option in Dict:
               break
            else:
                #把content丢进chatgpt函数里再滚一遍，获取新dict
                dict_content = chatgpt_options(Content)
                Dict.clear()
                for line in dict_content.split('\n'):
                   parts = line.split('. ')
                   Dict[parts[0]] = parts[1]
        user_message = Dict[option]#这里是跳出循环后的语句，将获得被选择的文本
        history.append([user_message, None])#这里将选定的文本更新到聊天记录里

        return history,dict_content

from urllib.request import urlopen
from urllib.request import Request
from urllib.error import URLError
from urllib.parse import urlencode
import time

timer = time.perf_counter
API_KEY = 'your API_KEY'#从百度控制台https://login.bce.baidu.com/获取key和secret
SECRET_KEY = 'your Secret_KEY'

# 需要识别的文件
# 只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
CUID = '19.00.2';
# 采样率
RATE = 16000;  # 固定值
# 普通版
DEV_PID = 1537;  # 1537 表示识别普通话，使用输入法模型。根据文档填写PID，选择语言及识别模型
ASR_URL = 'http://vop.baidu.com/server_api'
SCOPE = 'audio_voice_assistant_get'  # 有此scope表示有asr能力，没有请在网页里勾选，非常旧的应用可能没有
TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
class DemoError(Exception):
    pass
def fetch_token():
    params = {'grant_type': 'client_credentials',
              'client_id': API_KEY,
              'client_secret': SECRET_KEY}
    post_data = urlencode(params)
    post_data = post_data.encode('utf-8')
    req = Request(TOKEN_URL, post_data)
    try:
        f = urlopen(req)
        result_str = f.read()
    except URLError as err:
        print('token http response http code : ' + str(err.code))
        result_str = err.read()
    result_str = result_str.decode()

    print(result_str)
    result = json.loads(result_str)
    print(result)
    if ('access_token' in result.keys() and 'scope' in result.keys()):
        if SCOPE and (not SCOPE in result['scope'].split(' ')):  # SCOPE = False 忽略检查
            raise DemoError('scope is not correct')
        print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
        return result['access_token']
    else:
        raise DemoError('MAYBE API_KEY or SECRET_KEY not correct: access_token or scope not found in token response')

def resample_rate(path,new_sample_rate = 16000):

    signal, sr = librosa.load(path, sr=None)
    wavfile = path.split('/')[-1]
    wavfile = wavfile.split('.')[0]
    file_name = wavfile + '_new.wav'
    new_signal = librosa.resample(signal, sr, new_sample_rate) # 
    #librosa.output.write_wav(file_name, new_signal , new_sample_rate) 
    sf.write(file_name, new_signal, new_sample_rate, subtype='PCM_16')
    print(f'{file_name} has download.')
    return f'{file_name}'
def baiduapimain(file_name):
    #token = fetch_token()
    # 文件格式
    FORMAT = file_name[-3:];  # 文件后缀只支持 pcm/wav/amr 格式，极速版额外支持m4a 格式
    """
    httpHandler = urllib2.HTTPHandler(debuglevel=1)
    opener = urllib2.build_opener(httpHandler)
    urllib2.install_opener(opener)
    """
    # 将采样率转换为 16K
    file_name = resample_rate(file_name,new_sample_rate = 16000)
    speech_data = []
    with open(file_name, 'rb') as speech_file:
        speech_data = speech_file.read()
    length = len(speech_data)
    if length == 0:
        raise DemoError('file %s length read 0 bytes' % file_name)

    params = {'cuid': CUID, 'token': '24.02f321bc64cd9dacefe1fb1aad676033.2592000.1686489173.282335-32563874', 'dev_pid': DEV_PID}
    params_query = urlencode(params);

    headers = {
        'Content-Type': 'audio/' + FORMAT + '; rate=' + str(RATE),
        'Content-Length': length
    }

    url = ASR_URL + "?" + params_query
    print("url is", url);
    print("header is", headers)
    # print post_data
    req = Request(ASR_URL + "?" + params_query, speech_data, headers)
    try:
        begin = timer()
        f = urlopen(req)
        result_str = f.read()
        print("Request time cost %f" % (timer() - begin))
    except  URLError as err:
        print('asr http response http code : ' + str(err.code))
        result_str = err.read()

    result_str = str(result_str, 'utf-8')
    print(result_str)
    # 解析JSON字符串
    result_dict = json.loads(result_str)
    # 提取文本
    result_content = result_dict['result'][0]
    # 输出结果
    return result_content
download_counter = 0
def save_audio(audio_file,history):
    global download_counter
    download_counter += 1

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    file_name = f"{current_date}_第{download_counter}次录制.wav"
    # 将临时文件保存到代码文件目录中
    destination_path = os.path.join(os.getcwd(), file_name)
    shutil.copyfile(audio_file, destination_path)

    # audio_data = audio[1].astype(np.int16)
    # sample_rate = audio[0]
    # wav.write(file_name, sample_rate, audio_data)

    transcript = baiduapimain(destination_path)
    history.append([None, transcript])

    return transcript,history
  
def update_chat_history(history):
    return history

# event dialogue，事件对话，针对有固定对话模板的对话场景
with gr.Blocks() as eventdia_demo:  # gradio的块布局。docs:https://gradio.app/docs/#blocks
    history = gr.State([])  # docs:https://gradio.app/docs/#state
    Scene = gr.State("")
    Content = gr.State("")
    Dict = gr.State({})
    gr.Markdown(
        "Start typing below and then click **Run** to see the output.")  # 这是排头提示,docs:https://gradio.app/docs/#markdown
    with gr.Row():  # 这是行布局,docs:https://gradio.app/docs/#block-layouts
        
        with gr.Column(scale=3):  # 这是列布局,docs:https://gradio.app/docs/#block-layouts
            # 在这里添加需要的组件，例如输入框、按钮等
            chatbot = gr.Chatbot(elem_id="chatbot").style(height=450)
            dictionary_output = gr.Textbox( lines=5, label="回复选项",
                                           elem_id='dict_display')
            optionbox = gr.Radio(choices=options, elem_id="option_box")  # docs:https://gradio.app/docs/#radio
            with gr.Row():
              with gr.Column(scale=5):
                attitude = gr.Slider(minimum=0, maximum=10, value=attipredict, step=1, label="越大表示越同意,越小越反对,5为中立,请表明您的态度",interactive=True,elem_id="attitude")
              with gr.Column(scale=1):
                attitude_int = gr.Button("确定", elem_id='attitude_value')
            send_message = gr.Button("发送", elem_id="send_message_button")  # docs:https://gradio.app/docs/#button

        with gr.Column(scale=2):  # 这是列布局
            # 在这里添加需要的组件，例如输入框、按钮等
            imagepath = gr.Image(source="upload", type="filepath", interactive=True, elem_id="image_path").style(
                width="375px", height="375px")  # docs:https://gradio.app/docs/#image
            dialogbutton = gr.Button("识别图像并生成选项", elem_id="dialog_button")
            Contentdisplay = gr.Textbox(lines=4, label="文本展示", interactive=False,elem_id="Condisply")
            audiomsgdisplay = gr.Textbox(lines=4, label="回复内容", interactive=False, elem_id="audiomsgdis")
            chatresponse = gr.Audio(source="microphone", type="filepath", interactive=True,
                                    elem_id="chat_response",)  # docs:https://gradio.app/docs/#audio
            audiomsgbutton = gr.Button("记录对象回复", elem_id="audio_msg_button")
    dialogbutton.click(eventdia, inputs=[imagepath, history, Dict],
                       outputs=[dictionary_output, Contentdisplay])  # docs:https://gradio.app/docs/#button-click-header
    send_message.click(optionmsg, inputs=[optionbox, Dict, Content, history], outputs=[chatbot,dictionary_output])
    audiomsgbutton.click(fn=save_audio, inputs=[chatresponse, history], outputs=[audiomsgdisplay, chatbot])
    attitude_int.click(fn=attitudeoptions, inputs=[attitude,history],outputs=chatbot)

# %%
# 定义 chardia_demo
chardia_demo = gr.load(
    "huggingface/facebook/wav2vec2-base-960h",
    title=None,
    inputs="mic",
    description="Let me try to guess what you're saying!",
)

# %%
# 创建 TabbedInterface，将 eventdia_demo 和 chardia_demo 放置在选项卡中
interface_list = [eventdia_demo, chardia_demo]
tab_names = ["eventdia_demo", "chardia_demo"]
interface = gr.TabbedInterface(interface_list, tab_names=tab_names, title="My Demo App")

# 启动 Gradio 服务
interface.launch(share=True, show_error=True)


# %%

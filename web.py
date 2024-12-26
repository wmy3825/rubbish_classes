
import torch
import streamlit as st
from PIL import Image
from infer import infer
from models import AlexNet
from Alexnet_infer import Alexnet_infer
from models import Residual
import datetime
import requests
import base64

# 设置页面配置
st.set_page_config(
    page_title="智能垃圾分类系统展示平台",
    page_icon=":recycle:",
    layout="wide",
    initial_sidebar_state="auto"
)

# 自定义CSS样式，进行更全面细致的样式设计
def local_css(file_name):
    encodings = ['utf-8', 'gbk', 'latin-1']
    for encoding in encodings:
        try:
            with open(file_name, encoding=encoding) as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
            break
        except UnicodeDecodeError:
            continue
# 应用CSS样式
local_css("styles.css")
st.markdown("<h1 style='text-align: center; color: #3366ff;'>欢迎来到垃圾分类系统展示平台</h1>", unsafe_allow_html=True)
# 使用st.tabs创建导航标签页
tab1, tab2, tab3 = st.tabs(["首页", "功能展示",  "联系我们"])

# 获取当前的天气，添加获取天气图标对应代码（这里仅示意，需结合实际天气图标资源来完善）
def get_weather(city):
    api_url = f"http://wttr.in/{city}?format=%c+%t"
    response = requests.get(api_url)
    if response.status_code == 200:
        weather_text = response.text
        # 假设根据天气文本中的关键信息提取图标代码，这里简单示例，实际需更准确逻辑
        icon_code = "sunny" if "晴" in weather_text else "cloudy"  # 简单示例，需完善判断逻辑
        return weather_text, icon_code
    else:
        return "获取天气信息失败，请重试！", ""

# 获取当前日期和星期
now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d")
weekday = now.strftime("%A")

# 在侧边栏添加日期和星期显示
st.sidebar.markdown(f"<div style='text-align: left; margin-top: auto;'><p style='font-size: 14px;'>当前日期：{date}</p><p style='font-size: 14px;'>星期：{weekday}</p></div>", unsafe_allow_html=True)
st.sidebar.markdown(f"垃圾识别分类的背景主要包括以下几个方面：<ul><li>垃圾产生量的急剧增加</li><li>传统垃圾处理方式具有局限性</li><li>资源短缺与可持续发展需求</li><li>环境保护意识的提高</li></ul>", unsafe_allow_html=True)

# 在侧栏添加输入框和天气展示部分
city = st.sidebar.text_input("输入城市名称：", "承德")
if city:
    weather_info, weather_icon = get_weather(city)
    st.sidebar.subheader(f"{city}的实时天气：")
    st.sidebar.markdown(f"<span>{weather_info}</span> <i class='{weather_icon}-icon'></i>", unsafe_allow_html=True)  # 简单示意显示图标，需完善图标类名等

def generate_video_html(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    video_base64 = base64.b64encode(video_bytes).decode('utf-8')
    return f"""
    <video controls autoplay loop muted>
        <source type="video/mp4" src="data:video/mp4;base64,{video_base64}">
    </video>
    """


# 修正后的垃圾分类函数（保持不变，假设功能完整可用）
def classify_garbage(label):
    if label in ['充电宝', '易拉罐', '铝管药膏', '铝罐', '电线', '塑料桶', '塑料瓶', '塑料盆', '塑料衣架',
                 '塑料袋', '快递纸袋', '纸箱子', '药盒', '帆布包', '费衣服', '玻璃杯', '玩具', '皮鞋', '筷子']:
        return "可回收物", "可回收物是指适宜回收利用和资源化利用的生活废弃物。"
    elif label in ['电池', '洗护用品']:
        return "有害垃圾", "有害垃圾是指对人体健康或者自然环境造成直接或者潜在危害的生活废弃物。"
    elif label in ['果皮', '烂水果', '绿叶菜', '饭菜', '骨头', '鱼骨头', '鸡蛋壳', '茶叶', '调料']:
        return "厨余垃圾", "厨余垃圾是指居民日常生活及食品加工、饮食服务、单位供餐等活动中产生的垃圾。"
    elif label in ['烟头', '牙签', '陶瓷', '餐盒', '酒']:
        return "其他垃圾", "其他垃圾是指除可回收物、有害垃圾、厨余垃圾以外的其他生活废弃物。"
    else:
        return 'uncertain', ''

# 首页内容
with tab1:
    # 这里添加动画效果示例，假设嵌入一个简单HTML动画元素（需替换为实际可用的动画代码）
    video_path = '分类宣传视频.mp4'
    video_html = generate_video_html(video_path)
    st.markdown(video_html, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>这是一个集成了先进技术的智能分类系统，旨在为用户提供高效、便捷的服务。</p>", unsafe_allow_html=True)

# 功能展示
with tab2:
    model_type = st.selectbox('请选择你需要使用的模型', ['resnet专用模型', 'AlexNet模型'], key="model_select", help="选择用于垃圾分类的模型")  # 应用样式并添加提示信息
    up_image = st.file_uploader('请上传一张需要识别的垃圾图片.....')

    if up_image is not None:
        image = Image.open(up_image).convert('RGB')
        st.image(up_image)
        st.success("图片加载完毕")
        if model_type == 'resnet专用模型':
            torch.serialization.add_safe_globals({'Residual': Residual})
            model_file = 'model/trainresnet.pth'
            labels = infer(image, model_file)
        elif model_type == 'AlexNet模型':
            torch.serialization.add_safe_globals({'Residual': AlexNet})
            model_file = 'model/trainalexnet.pth'
            labels = Alexnet_infer(image, model_file)
        if labels:
            for label, prob in labels:
                category, description = classify_garbage(label)
                st.markdown(f"""
                    <div class='classification-result'>
                        <p><b>预测类别：</b> {label}</p>
                        <p><b>概率：</b> {prob}</p>
                        <p><b>垃圾所属类别：</b> {category}</p>
                        <p><b>类别描述：</b> {description}</p>
                    </div>
                    """, unsafe_allow_html=True)
                st.balloons()

# 图片搜索页面
with tab3:
    st.markdown("<h2 class='case-title'>联系我们</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='contact-card'>
        <i class='phone-icon'></i><span><b>电话：</b> 888-8888</span>
    </div>
    <div class='contact-card'>
        <i class='email-icon'></i><span><b>邮箱：</b> 666@qq.com</span>
    </div>
    <div class='contact-card'>
        <i class='location-icon'></i><span><b>地址：</b> 河北民师</span>
    </div>
    """, unsafe_allow_html=True)  # 展示联系方式，应用样式并添加图标（需完善图标类名等）

# 页脚
st.markdown("<footer style='text-align: center; color: #999; margin-top: 50px;'>© [智慧星辰团] 版权所有</footer>", unsafe_allow_html=True)








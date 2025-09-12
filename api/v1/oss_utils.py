"""
OSS相关工具函数
"""
import os
import io
import uuid
import logging
from typing import Optional
from PIL import Image
import oss2
from urllib.parse import urlparse
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ----------------- 阿里云 OSS 配置 -----------------
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID')
ACCESS_KEY_SECRET = os.getenv('ACCESS_KEY_SECRET')
ENDPOINT = os.getenv('ENDPOINT', 'oss-cn-shanghai.aliyuncs.com')
BUCKET_NAME = os.getenv('BUCKET_NAME')

# 检查必要配置是否存在
if not ACCESS_KEY_ID or not ACCESS_KEY_SECRET or ACCESS_KEY_ID == "your_actual_access_key_id_here":
    raise ValueError("请在.env文件中配置正确的阿里云ACCESS_KEY_ID和ACCESS_KEY_SECRET")

# 创建 OSS 客户端
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

# 创建logger对象
logger = logging.getLogger(__name__)


def upload_image_to_oss(img: Image.Image, bucket, request_id: str = None, file_type: str = "origin") -> str:
    """
    上传图片到阿里云OSS
    """
    # 如果图像是 RGBA 模式，先转换为 RGB 模式
    if img.mode == 'RGBA':
        img = img.convert('RGB')  # 转换为 RGB 模式

    file_id = str(uuid.uuid4()) + ".jpg"
    
    # 根据request_id和文件类型组织文件夹结构
    if request_id:
        if file_type == "origin":
            file_path = f"input_face/{request_id}/origin_face/{file_id}"
        else:  # cropped or flipped
            file_path = f"input_face/{request_id}/corp_face/{file_id}"
    else:
        file_path = file_id
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 上传到 OSS
    bucket.put_object(file_path, img_bytes)
    
    # 构建图片 URL
    image_url = f"http://{BUCKET_NAME}.{ENDPOINT}/{file_path}"
    return image_url


def save_description_to_oss(description: str, image_url: str, face_request_id: str, bucket):
    """
    将描述保存为txt文件并上传到OSS
    """
    try:
        # 移除<|begin_of_box|>和<|end_of_box|>标记
        cleaned_description = description
        if "<|begin_of_box|>" in cleaned_description:
            cleaned_description = cleaned_description.replace("<|begin_of_box|>", "")
        if "<|end_of_box|>" in cleaned_description:
            cleaned_description = cleaned_description.replace("<|end_of_box|>", "")
        
        # 从image_url中提取文件名（不含扩展名）
        parsed_url = urlparse(image_url)
        file_name = parsed_url.path.split('/')[-1]  # 获取文件名
        file_name_without_ext = '.'.join(file_name.split('.')[:-1]) if '.' in file_name else file_name
        
        # 构建OSS路径
        oss_path = f"input_face/{face_request_id}/per_face&prompt/{file_name_without_ext}.txt"
        
        # 将描述内容转换为字节流
        description_bytes = cleaned_description.encode('utf-8')
        
        # 上传到OSS
        bucket.put_object(oss_path, description_bytes)
        
        logger.info(f"描述文件已保存到OSS: {oss_path}")
    except Exception as e:
        logger.error(f"保存描述文件到OSS时出错: {e}")
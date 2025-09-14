"""
人脸处理相关功能
"""
import os
import io
import uuid
import base64
import logging
from typing import Tuple, List, Optional
from PIL import Image
from crop_face import FaceCropper
from .oss_utils import upload_image_to_oss
from .models import FaceResult

logger = logging.getLogger(__name__)

# 获取项目根目录路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------- 公共工具 -----------------
def _parse_wh_tuple(s: str, default: Tuple[int, int]) -> Tuple[int, int]:
    try:
        w, h = map(int, s.split(","))
        return (w, h) if w > 0 and h > 0 else default
    except Exception:
        return default

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _save_img_to_static(img: Image.Image) -> str:
    file_id = str(uuid.uuid4())
    save_path = f"static/{file_id}.jpg"
    img.save(save_path, format="JPEG")
    return f"/static/{file_id}.jpg"

def save_or_encode(img: Image.Image, return_type: str):
    return _pil_to_b64(img) if return_type == "base64" else _save_img_to_static(img)

def process_face_images(files: List, request_id: str, crop_size_tuple: Tuple[int, int], min_size_tuple: Tuple[int, int],
                       expand_ratio: float, mark_faces: bool, largest_face_only: bool, save_flipped: bool,
                       face_detector, strict_face_filter: bool) -> tuple[List[FaceResult], int, int]:
    """
    处理上传的人脸图片
    """
    # 构建模型路径
    model_path = os.path.join(PROJECT_ROOT, "models", "face_yolov8m.pt")
    
    # 检查模型文件是否存在
    if face_detector == "yolov8" and not os.path.exists(model_path):
        logger.warning(f"YOLO模型文件不存在: {model_path}")
        # 可以选择使用默认模型或抛出异常
        model_path = None
    
    cropper = FaceCropper(
        min_neighbors=8,  # 使用默认值
        scale_factor=1.3,  # 使用默认值
        min_size=min_size_tuple,
        confidence_threshold=0.5,  # 使用默认值
        use_mtcnn=(face_detector == "mtcnn"),
        use_yolov8=(face_detector == "yolov8"),
        use_blazeface=(face_detector == "blazeface"),
        use_ultraface=(face_detector == "ultraface"),
        strict_face_filter=strict_face_filter
    )

    results: List[FaceResult] = []
    
    # 统计输入和输出的图片数量
    input_image_count = len(files)
    output_face_count = 0

    for file in files:
        # 注意：这里需要在异步环境中正确处理文件读取
        contents = file.file.read()
        path = f"temp_{file.filename}"
        with open(path, "wb") as f:
            f.write(contents)

        try:
            # 上传原始图片到阿里云 OSS
            original_img = Image.open(io.BytesIO(contents))
            input_image_url = upload_image_to_oss(original_img, upload_image_to_oss.__globals__['bucket'], request_id, "origin")
            
            cropped_img, flipped_img, marked_img = cropper.crop_largest_face_in_image(
                path, None, crop_size_tuple, expand_ratio, largest_face_only
            )

            if cropped_img:
                # 上传裁剪后的图片到阿里云 OSS
                image_url = upload_image_to_oss(cropped_img, upload_image_to_oss.__globals__['bucket'], request_id, "cropped")
                output_face_count += 1

                # 上传水平翻转的图片到 OSS
                flipped_image_url = None
                if save_flipped and flipped_img:
                    flipped_image_url = upload_image_to_oss(flipped_img, upload_image_to_oss.__globals__['bucket'], request_id, "flipped")
                    output_face_count += 1

                # 上传标记人脸的图片到 OSS
                marked_image_url = None
                if mark_faces and marked_img:
                    marked_image_url = upload_image_to_oss(marked_img, upload_image_to_oss.__globals__['bucket'], request_id, "marked")

                # 根据返回的URL生成结果
                results.append(FaceResult(
                    filename=file.filename,
                    input_image_url=input_image_url,
                    images_url=[image_url],
                    flipped_images_url=[flipped_image_url] if save_flipped and flipped_img else None,
                    marked_image_url=marked_image_url if mark_faces and marked_img else None,
                    images_base64=None,
                    flipped_images_base64=None,
                    marked_image_base64=None,
                    error=None
                ))
            else:
                results.append(FaceResult(
                    filename=file.filename,
                    input_image_url=input_image_url,
                    images_url=None,
                    flipped_images_url=None,
                    marked_image_url=None,
                    images_base64=None,
                    flipped_images_base64=None,
                    marked_image_base64=None,
                    error="No face detected"
                ))
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    return results, input_image_count, output_face_count
import os
import io
import re
import uuid
import oss2
import time
import base64
import asyncio
import json
import concurrent.futures
import logging
from enum import Enum
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, APIRouter, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Column, String, Float, Text, Enum as SqlEnum, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from crop_face import FaceCropper
from zhipuai import ZhipuAI

# ----------------- 阿里云 OSS 配置 -----------------
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID', 'LTAI5t7Do4fXYVDevinHu9yN')
ACCESS_KEY_SECRET = os.getenv('ACCESS_KEY_SECRET', 'AIe6KrPQQHBw17ecfBVFi1jdNrdcVy')
ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'  # 替换为你实际的 OSS 访问域名
BUCKET_NAME = 'drawbookai'

# 创建 OSS 客户端
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

# 创建logger对象
logger = logging.getLogger(__name__)

class TaskStatus(str,Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"

# ---------- SQLAlchemy 初始化 ----------
Base = declarative_base()

class Task(Base):
    __tablename__ = "tasks"

    request_id = Column(String(64), primary_key=True, index=True)
    type = Column(String(32), index=True)              # faces / glm
    status = Column(SqlEnum(TaskStatus), index=True)
    created_at = Column(Float, index=True)
    finished_at = Column(Float, nullable=True)
    duration = Column(Float, nullable=True)
    model = Column(String(64), nullable=True)
    results = Column(Text)  # JSON 序列化结果

    __table_args__ = (
        Index("idx_type_status", "type", "status"),
    )

# MySQL 连接串
MYSQL_URL = "mysql+pymysql://drawbook:WwY2fjcMCDyjDDxJ@139.196.51.51:3306/drawbook"

engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))

# 创建表
Base.metadata.create_all(engine)

# ----------------- FastAPI 初始化 -----------------
app = FastAPI(
    title="Draw Book AI API",
    description="脸模训练相关API接口",
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/drawbook-ai.json",
)

# 挂载静态目录
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# ----------------- 工具函数 -----------------
def save_task(request_id, task_type, status, created_at, finished_at=None, duration=None, model=None, results=None):
    db = SessionLocal()
    try:
        if results:
            # 自动兼容 Pydantic 模型和 dict
            results = [r.dict() if hasattr(r, "dict") else r for r in results]
        task = Task(
            request_id=request_id,
            type=task_type,
            status=status,
            created_at=created_at,
            finished_at=finished_at,
            duration=duration,
            model=model,
            results=json.dumps(results, ensure_ascii=False) if results else None,
        )
        db.merge(task)
        db.commit()
    finally:
        db.close()
def get_task(request_id):
    db = SessionLocal()
    try:
        return db.query(Task).filter(Task.request_id == request_id).first()
    finally:
        db.close()

def list_tasks(page: int = 1, size: int = 20):
    db = SessionLocal()
    try:
        return db.query(Task).order_by(Task.created_at.desc()).offset((page - 1) * size).limit(size).all()
    finally:
        db.close()

def cleanup_expired_tasks(expire_seconds: int = 7 * 24 * 3600):
    now = time.time()
    db = SessionLocal()
    try:
        db.query(Task).filter(Task.created_at < now - expire_seconds).delete()
        db.commit()
    finally:
        db.close()

def build_absolute_url(path: str, request: Request) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path  # 如果已经是完整的 URL，直接返回
    base_url = str(request.base_url).rstrip("/")  # 获取基础 URL，确保没有多余的 "/"
    # 拼接成完整的 URL，不重复添加 http://
    return f"http://{BUCKET_NAME}.{ENDPOINT}/{path.lstrip('/')}"

# ----------------- 数据模型 -----------------

class FaceResult(BaseModel):
    filename: str
    input_image_url: Optional[str] = None
    images_url: Optional[List[str]] = None
    flipped_images_url: Optional[List[str]] = None
    marked_image_url: Optional[str] = None
    images_base64: Optional[List[str]] = None
    flipped_images_base64: Optional[List[str]] = None
    marked_image_base64: Optional[str] = None
    error: Optional[str] = None

class FaceDetectionAlgorithm(str, Enum):
    yolov8 = "yolov8"
    mtcnn = "mtcnn"
    blazeface = "blazeface"
    ultraface = "ultraface"
    opencv_dnn = "opencv_dnn"
    haar_cascade = "haar_cascade"

class MultiFaceResponse(BaseModel):
    request_id: str
    status: TaskStatus
    created_at: float
    finished_at: float
    duration: float
    input_num: int
    output_num: int
    results: List[FaceResult]

class StatusResponse(BaseModel):
    request_id: str
    type: str
    status: TaskStatus
    created_at: float
    finished_at: float
    duration: float
    model: Optional[str] = None
    results: Optional[List[FaceResult]] = None

class GLMItemResult(BaseModel):
    image_url: str
    status: TaskStatus
    description: Optional[str] = None
    error: Optional[str] = None

class GLMBatchResponse(BaseModel):
    request_id: str
    status: TaskStatus
    created_at: float
    finished_at: float
    duration: float
    model: str
    total: int
    results: List[GLMItemResult]

class GLMStatusResponse(BaseModel):
    request_id: str
    type: str
    status: TaskStatus
    created_at: float
    finished_at: float
    duration: float
    model: Optional[str] = None
    total: Optional[int] = None
    results: Optional[List[GLMItemResult]] = None

class ReturnType(str, Enum):
    url = "url"

# ----------------- 阿里云 OSS 上传 -----------------
def upload_image_to_oss(img: Image.Image, bucket, request_id: str = None, file_type: str = "origin") -> str:
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

# ----------------- ZhipuAI 工具 -----------------
client = None
def init_zhipu_client(api_key: str = None) -> bool:
    global client
    try:
        key = api_key or os.getenv("ZHIPUAI_API_KEY")
        if not key:
            raise ValueError("API Key 未提供")
        client = ZhipuAI(api_key=key)
        return True
    except Exception as e:
        print(f"初始化 ZhipuAI 客户端失败: {e}")
        client = None
        return False

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename or "image")

def save_description_to_file(image_url: str, description: str):
    parsed = urlparse(image_url)
    base_name = os.path.basename(parsed.path) or "image"
    file_name = sanitize_filename(os.path.splitext(base_name)[0])
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{file_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(description)

async def process_batch(urls: List[str], prompt: str, model: str) -> List[dict]:
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, fetch_image_description_sync, u, prompt, model) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r if isinstance(r, dict) else {"status": "error", "error": str(r)} for r in results]

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
        from urllib.parse import urlparse
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

def fetch_image_description_sync(image_url: str, prompt: str, model: str) -> dict:
    if not client:
        return {"image_url": image_url, "status": TaskStatus.failed, "error": "ZhipuAI 未初始化"}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            temperature=0.5,
            max_tokens=2048,
        )
        description = response.choices[0].message.content
        return {"image_url": image_url, "status": TaskStatus.completed, "description": description}
    except Exception as e:
        return {"image_url": image_url, "status": TaskStatus.failed, "error": str(e)}

# ----------------- API 路由 -----------------
router_v1 = APIRouter(prefix="/api/v1", tags=["drawbook-ai"])

# 1) 人脸裁剪接口
@router_v1.post(
        "/get_faces", 
        operation_id="get_faces",
        response_model=MultiFaceResponse
)
async def get_faces(
    files: List[UploadFile] = File(..., description="上传的图片文件，支持多文件"),
    crop_size: str = Form("1024,1024", description="裁切尺寸，格式：宽,高"),
    expand_ratio: float = Form(1.9, description="扩展比例"),
    mark_faces: bool = Form(False, description="是否返回标记人脸的原图"),
    largest_face_only: bool = Form(True, description="是否只裁剪最大的人脸"),
    min_neighbors: int = Form(8, description="人脸检测的 minNeighbors 参数"),
    scale_factor: float = Form(1.3, description="人脸检测的 scaleFactor 参数"),
    min_size: str = Form("60,60", description="人脸检测的 minSize 参数，格式：宽,高"),
    save_flipped: bool = Form(True, description="是否保存水平翻转的人脸"),
    confidence_threshold: float = Form(0.5, description="人脸检测的置信度阈值"),
    face_detector: FaceDetectionAlgorithm = Form(FaceDetectionAlgorithm.yolov8, description="人脸检测算法"),
    strict_face_filter: bool = Form(False, description="是否使用严格的人脸面积过滤"),
):
    request_id = str(uuid.uuid4())
    created_at = time.time()

    crop_size_tuple = _parse_wh_tuple(crop_size, (512, 512))
    min_size_tuple = _parse_wh_tuple(min_size, (30, 30))

    cropper = FaceCropper(
        min_neighbors=min_neighbors,
        scale_factor=scale_factor,
        min_size=min_size_tuple,
        confidence_threshold=confidence_threshold,
        use_mtcnn=(face_detector == FaceDetectionAlgorithm.mtcnn),
        use_yolov8=(face_detector == FaceDetectionAlgorithm.yolov8),
        use_blazeface=(face_detector == FaceDetectionAlgorithm.blazeface),
        use_ultraface=(face_detector == FaceDetectionAlgorithm.ultraface),
        strict_face_filter=strict_face_filter,
    )

    results: List[FaceResult] = []
    
    # 统计输入和输出的图片数量
    input_image_count = len(files)
    output_face_count = 0

    for file in files:
        contents = await file.read()
        path = f"temp_{file.filename}"
        with open(path, "wb") as f:
            f.write(contents)

        try:
            # 上传原始图片到阿里云 OSS
            original_img = Image.open(io.BytesIO(contents))
            input_image_url = upload_image_to_oss(original_img, bucket, request_id, "origin")
            
            cropped_img, flipped_img, marked_img = cropper.crop_largest_face_in_image(
                path, None, crop_size_tuple, expand_ratio, largest_face_only
            )

            if cropped_img:
                # 上传裁剪后的图片到阿里云 OSS
                image_url = upload_image_to_oss(cropped_img, bucket, request_id, "cropped")
                output_face_count += 1

                # 上传水平翻转的图片到 OSS
                flipped_image_url = None
                if save_flipped and flipped_img:
                    flipped_image_url = upload_image_to_oss(flipped_img, bucket, request_id, "flipped")
                    output_face_count += 1

                # 上传标记人脸的图片到 OSS
                marked_image_url = None
                if mark_faces and marked_img:
                    marked_image_url = upload_image_to_oss(marked_img, bucket, request_id, "marked")

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

    finished_at = time.time()
    duration = finished_at - created_at
    save_task(request_id, "faces", TaskStatus.completed, created_at, finished_at, duration, results=results)
    
    # 记录输入和输出图片数量
    logger.info(f"任务 {request_id} 完成: 输入图片 {input_image_count} 张，输出人脸图片 {output_face_count} 张")

    return MultiFaceResponse(
        request_id=request_id,
        status=TaskStatus.completed,
        created_at=created_at,
        finished_at=finished_at,
        duration=duration,
        input_num=input_image_count,
        output_num=output_face_count,
        results=results
    )

# 查询任务
@router_v1.get("/get_task/{request_id}")
def get_task_api(request_id: str):
    task = get_task(request_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 根据任务类型返回不同的响应模型
    results = json.loads(task.results) if task.results else None
    
    if task.type == "glm":
        # GLM任务使用GLMStatusResponse模型
        return GLMStatusResponse(
            request_id=task.request_id,
            type=task.type,
            status=task.status,
            created_at=task.created_at,
            finished_at=task.finished_at,
            duration=task.duration,
            model=task.model,
            total=len(results) if results else 0,
            results=[GLMItemResult(**result) for result in results] if results else None,
        )
    else:
        # 人脸检测任务使用StatusResponse模型
        # 如果是GLM任务，需要转换结果格式以匹配FaceResult模型
        if task.type == "glm" and results:
            # 将GLM结果转换为FaceResult格式
            converted_results = []
            for result in results:
                converted_result = {
                    "filename": result.get("image_url", "").split("/")[-1] or "unknown",
                    "images_url": [result.get("image_url")] if result.get("image_url") else None,
                    "flipped_images_url": None,
                    "marked_image_url": None,
                    "input_image_url": result.get("image_url"),
                    "images_base64": None,
                    "flipped_images_base64": None,
                    "marked_image_base64": None,
                    "error": result.get("error")
                }
                converted_results.append(converted_result)
            results = converted_results
        
        return StatusResponse(
            request_id=task.request_id,
            type=task.type,
            status=task.status,
            created_at=task.created_at,
            finished_at=task.finished_at,
            duration=task.duration,
            model=task.model,
            results=results,
        )

# 分页查询任务
@router_v1.get("/list_tasks")
def list_tasks_api(page: int = 1, size: int = 20):
    tasks = list_tasks(page, size)
    return [{
        "request_id": t.request_id,
        "type": t.type,
        "status": t.status.value,
        "created_at": t.created_at,
        "finished_at": t.finished_at,
        "duration": t.duration,
    } for t in tasks]

# ----------------- GLM 批处理接口 -----------------
@router_v1.post(
        "/glm_batch_prompt",
        operation_id="glm_batch_prompt",
        response_model=GLMBatchResponse
        )
async def glm_batch_prompt(
    api_key: str = Form(..., description="GLM API Key"),
    face_request_id: str = Form(None, description="人脸处理任务的request_id"),
    image_url: Optional[str] = Form(None, description="图片 URL（逗号分隔）"),
    prompt: str = Form(..., description="提示词"),
    model: str = Form("glm-4.5v", description="模型"),
):
    request_id = str(uuid.uuid4())
    created_at = time.time()

    if not init_zhipu_client(api_key):
        save_task(request_id, "glm", TaskStatus.failed, created_at)
        raise HTTPException(status_code=500, detail="ZhipuAI 初始化失败")

    # 获取图片URL列表
    urls: List[str] = []
    
    # 如果提供了face_request_id，则从任务中获取图片URL
    if face_request_id:
        task = get_task(face_request_id)
        if task and task.results:
            try:
                results = json.loads(task.results)
                # 从结果中提取所有images_url和flipped_images_url
                for result in results:
                    if "images_url" in result and result["images_url"]:
                        urls.extend(result["images_url"])
                    if "flipped_images_url" in result and result["flipped_images_url"]:
                        urls.extend(result["flipped_images_url"])
            except Exception as e:
                logger.error(f"解析任务结果时出错: {e}")
    
    # 如果提供了image_url参数，则添加到URL列表中
    if image_url:
        urls.extend([u.strip() for u in image_url.split(",") if u.strip()])

    if not urls:
        save_task(request_id, "glm", TaskStatus.failed, created_at)
        raise HTTPException(status_code=400, detail="没有可处理的图片 URL")

    results_data = await process_batch(urls, prompt, model)
    
    # 保存描述到OSS
    if face_request_id:
        for result in results_data:
            if isinstance(result, dict) and result.get("status") == TaskStatus.completed and result.get("description"):
                try:
                    save_description_to_oss(result["description"], result["image_url"], face_request_id, bucket)
                except Exception as e:
                    logger.error(f"保存描述文件时出错: {e}")

    results: List[GLMItemResult] = [
        GLMItemResult(image_url=r.get("image_url", ""), status=r.get("status", TaskStatus.failed),
                      description=r.get("description"), error=r.get("error")) for r in results_data
    ]

    finished_at = time.time()
    duration = finished_at - created_at
    save_task(request_id, "glm", TaskStatus.completed, created_at, finished_at, duration, model=model,
              results=results)

    return GLMBatchResponse(request_id=request_id, status=TaskStatus.completed,
                            created_at=created_at, finished_at=finished_at,
                            duration=duration, model=model,
                            total=len(results), results=results)
app.include_router(router_v1)
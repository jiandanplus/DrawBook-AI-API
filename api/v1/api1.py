import os
import io
import uuid
import time
import base64
import asyncio
import json
import requests
import concurrent.futures
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
import oss2

# ----------------- 阿里云 OSS 配置 -----------------
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID', 'LTAI5t7Do4fXYVDevinHu9yN')
ACCESS_KEY_SECRET = os.getenv('ACCESS_KEY_SECRET', 'AIe6KrPQQHBw17ecfBVFi1jdNrdcVy')
ENDPOINT = 'oss-cn-shanghai.aliyuncs.com'  # 替换为你实际的 OSS 访问域名
BUCKET_NAME = 'drawbookai'

# 创建 OSS 客户端
auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)

# ----------------- 枚举定义 -----------------
class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"

# ----------------- SQLAlchemy 配置 -----------------
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

MYSQL_URL = "mysql+pymysql://drawbook:WwY2fjcMCDyjDDxJ@139.196.51.51:3306/drawbook"
engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))
Base.metadata.create_all(engine)

# ----------------- 工具函数 -----------------
def save_task(request_id, task_type, status, created_at, finished_at=None, duration=None, model=None, results=None):
    db = SessionLocal()
    try:
        if results:
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
    images_base64: Optional[List[str]] = None
    images_url: Optional[List[str]] = None
    error: Optional[str] = None

class MultiFaceResponse(BaseModel):
    request_id: str
    status: TaskStatus
    created_at: float
    finished_at: Optional[float]
    duration: Optional[float]
    results: List[FaceResult]

class StatusResponse(BaseModel):
    request_id: str
    type: str
    status: TaskStatus
    created_at: float
    finished_at: Optional[float]
    duration: Optional[float]
    model: Optional[str] = None
    results: Optional[List] = None

class GLMItemResult(BaseModel):
    image_url: str
    status: TaskStatus
    description: Optional[str] = None
    error: Optional[str] = None

class GLMBatchResponse(BaseModel):
    request_id: str
    status: TaskStatus
    created_at: float
    finished_at: Optional[float]
    duration: Optional[float]
    model: str
    total: int
    results: List[GLMItemResult]

class ReturnType(str, Enum):
    base64 = "base64"
    url = "url"

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

async def process_batch(urls: List[str], prompt: str, model: str) -> List[dict]:
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tasks = [loop.run_in_executor(executor, fetch_image_description_sync, u, prompt, model) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r if isinstance(r, dict) else {"status": TaskStatus.failed, "error": str(r)} for r in results]

# ----------------- 阿里云 OSS 上传 -----------------
def upload_image_to_oss(img: Image.Image, bucket) -> str:
    # 如果图像是 RGBA 模式，先转换为 RGB 模式
    if img.mode == 'RGBA':
        img = img.convert('RGB')  # 转换为 RGB 模式

    file_id = str(uuid.uuid4()) + ".jpg"
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # 上传到 OSS
    bucket.put_object(file_id, img_bytes)
    
    # 构建图片 URL
    image_url = f"http://{BUCKET_NAME}.{ENDPOINT}/{file_id}"
    return image_url


# ----------------- FastAPI 初始化 -----------------
app = FastAPI(
    title="Draw Book AI API",
    description="人脸检测与裁剪 + GLM 批处理（MySQL 版）",
    version="9.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
)

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ----------------- API 路由 -----------------
router_v1 = APIRouter(prefix="/api/v1", tags=["drawbook-ai"])

# ----------------- 数据处理 -----------------
@router_v1.post("/get_faces", response_model=MultiFaceResponse)
async def get_faces(
    request: Request, 
    files: List[UploadFile] = File(...), 
    return_type: ReturnType = Form(ReturnType.base64)
    ):
    request_id = str(uuid.uuid4())
    created_at = time.time()
    save_task(request_id, "faces", TaskStatus.running, created_at)

    cropper = FaceCropper()
    results: List[FaceResult] = []

    for file in files:
        contents = await file.read()
        path = f"temp_{file.filename}"
        with open(path, "wb") as f:
            f.write(contents)
        try:
            cropped_img, _ = cropper.crop_largest_face_in_image(path, None, (512, 512), 1.9, False)
            if cropped_img:
                image_url = upload_image_to_oss(cropped_img, bucket)
                results.append(FaceResult(
                    filename=file.filename,
                    images_base64=[None] if return_type == "base64" else None,
                    images_url=[image_url] if return_type == "url" else None
                ))
            else:
                results.append(FaceResult(filename=file.filename, error="No face detected"))
        finally:
            if os.path.exists(path):
                os.remove(path)

    finished_at = time.time()
    duration = finished_at - created_at
    save_task(request_id, "faces", TaskStatus.completed, created_at, finished_at, duration, results=results)

    return MultiFaceResponse(
        request_id=request_id,
        status=TaskStatus.completed,
        created_at=created_at,
        finished_at=finished_at,
        duration=duration,
        results=results
    )

# 挂载路由
app.include_router(router_v1)
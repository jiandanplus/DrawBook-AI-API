import os
import io
import re
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
from fastapi import FastAPI, UploadFile, File, Form, APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Column, String, Float, Text, Enum as SqlEnum, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from crop_face import FaceCropper
from zhipuai import ZhipuAI

# ---------- SQLAlchemy 初始化 ----------
Base = declarative_base()

class TaskStatus(str,Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"

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
    version="5.0.0",
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

def build_absolute_url(path: str, base_url: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{base_url.rstrip('/')}{path}"

# ----------------- 数据模型 -----------------

class FaceResult(BaseModel):
    filename: str
    images_base64: Optional[List[str]] = None
    flipped_images_base64: Optional[List[str]] = None
    marked_image_base64: Optional[str] = None
    images_url: Optional[List[str]] = None
    flipped_images_url: Optional[List[str]] = None
    marked_image_url: Optional[str] = None
    error: Optional[str] = None

class MultiFaceResponse(BaseModel):
    request_id: str
    status: TaskStatus
    created_at: float
    finished_at: float
    duration: float
    results: List[FaceResult]

class StatusResponse(BaseModel):
    request_id: str
    status: TaskStatus
    created_at: float
    finished_at: float
    duration: float
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
    return [r if isinstance(r, dict) else {"status": "error", "error": str(r)} for r in results]

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
    min_neighbors: int = Form(5, description="人脸检测的 minNeighbors 参数"),
    scale_factor: float = Form(1.3, description="人脸检测的 scaleFactor 参数"),
    min_size: str = Form("30,30", description="人脸检测的 minSize 参数，格式：宽,高"),
    save_flipped: bool = Form(True, description="是否保存水平翻转的人脸"),
    confidence_threshold: float = Form(0.7, description="人脸检测的置信度阈值"),
    return_type: ReturnType = Form(ReturnType.base64, description="返回类型，base64 或 url"),
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
    )

    results: List[FaceResult] = []

    for file in files:
        contents = await file.read()
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(contents)

        try:
            marked_val = None
            if mark_faces:
                marked_img = cropper.visualize_faces(image_path, output_path=None, show_largest_only=largest_face_only)
                if marked_img is not None:
                    marked_val = save_or_encode(marked_img, return_type)

            images_vals: List[str] = []
            flipped_vals: List[str] = []

            if largest_face_only:
                cropped_img, flipped_img = cropper.crop_largest_face_in_image(
                    image_path, None, crop_size_tuple, expand_ratio, save_flipped
                )
                if cropped_img is None:
                    results.append(FaceResult(filename=file.filename, error="No face detected"))
                else:
                    images_vals.append(save_or_encode(cropped_img, return_type))
                    if save_flipped:
                        flipped_img = flipped_img or cropped_img.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped_vals.append(save_or_encode(flipped_img, return_type))
                    results.append(
                        FaceResult(
                            filename=file.filename,
                            images_base64=images_vals if return_type == "base64" else None,
                            flipped_images_base64=flipped_vals if return_type == "base64" else None,
                            marked_image_base64=marked_val if return_type == "base64" else None,
                            images_url=images_vals if return_type == "url" else None,
                            flipped_images_url=flipped_vals if return_type == "url" else None,
                            marked_image_url=marked_val if return_type == "url" else None,
                        )
                    )
            else:
                cropped_list = cropper.crop_all_faces_in_image(image_path, None, crop_size_tuple, expand_ratio, False)
                if not cropped_list:
                    results.append(FaceResult(filename=file.filename, error="No face detected"))
                else:
                    for img in cropped_list:
                        images_vals.append(save_or_encode(img, return_type))
                        if save_flipped:
                            flipped_vals.append(save_or_encode(img.transpose(Image.FLIP_LEFT_RIGHT), return_type))
                    results.append(
                        FaceResult(
                            filename=file.filename,
                            images_base64=images_vals if return_type == "base64" else None,
                            flipped_images_base64=flipped_vals if return_type == "base64" else None,
                            images_url=images_vals if return_type == "url" else None,
                            flipped_images_url=flipped_vals if return_type == "url" else None,
                        )
                    )
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    finished_at = time.time()
    duration = finished_at - created_at
    save_task(request_id, "faces", TaskStatus.completed, created_at, finished_at, duration, results=results)

    return MultiFaceResponse(
        request_id=request_id,
        status="completed",
        created_at=created_at,
        finished_at=finished_at,
        duration=duration,
        results=results,
    )

# 查询任务
@router_v1.get("/get_task/{request_id}", response_model=StatusResponse)
def get_task_api(request_id: str):
    task = get_task(request_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return StatusResponse(
        request_id=task.request_id,
        type=task.type,
        status=task.status,
        created_at=task.created_at,
        finished_at=task.finished_at,
        duration=task.duration,
        model=task.model,
        results=json.loads(task.results) if task.results else None,
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
@router_v1.post("/glm_batch_prompt", response_model=GLMBatchResponse)
async def glm_batch_prompt(
    api_key: str = Form(..., description="GLM API Key"),
    image_url: Optional[str] = Form(None, description="图片 URL（逗号分隔）"),
    prompt: str = Form(..., description="提示词"),
    model: str = Form("glm-4.5v", description="模型"),
):
    request_id = str(uuid.uuid4())
    created_at = time.time()

    if not init_zhipu_client(api_key):
        save_task(request_id, "glm", TaskStatus.failed, created_at)
        raise HTTPException(status_code=500, detail="ZhipuAI 初始化失败")

    urls: List[str] = []
    if image_url:
        urls.extend([u.strip() for u in image_url.split(",") if u.strip()])

    if not urls:
        save_task(request_id, "glm", TaskStatus.failed, created_at)
        raise HTTPException(status_code=400, detail="没有可处理的图片 URL")

    results_data = await process_batch(urls, prompt, model)
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

# 挂载路由
app.include_router(router_v1)

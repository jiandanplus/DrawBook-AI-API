import os
import io
import uuid
import time
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, APIRouter, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .models import (
    FaceDetectionAlgorithm,
    ReturnType,
    MultiFaceResponse,
    StatusResponse,
    GLMStatusResponse,
    GLMItemResult,
    GLMBatchResponse,
    TaskStatus
)
from .database import save_task, get_task, list_tasks
from .oss_utils import bucket
from .face_processing import _parse_wh_tuple, process_face_images
from .glm_processing import init_zhipu_client, process_batch, get_image_urls_from_face_task, save_descriptions_to_oss

# 创建logger对象
logger = logging.getLogger(__name__)

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

    results, input_image_count, output_face_count = process_face_images(
        files=files,
        request_id=request_id,
        crop_size_tuple=crop_size_tuple,
        min_size_tuple=min_size_tuple,
        expand_ratio=expand_ratio,
        mark_faces=mark_faces,
        largest_face_only=largest_face_only,
        save_flipped=save_flipped,
        face_detector=face_detector.value,
        strict_face_filter=strict_face_filter
    )

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
@router_v1.get(
        "/get_task/{request_id}",
        operation_id="get_task"
        )
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
@router_v1.get(
        "/list_tasks",
        operation_id="list_tasks"
        )
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
    urls.extend(get_image_urls_from_face_task(face_request_id))
    
    # 如果提供了image_url参数，则添加到URL列表中
    if image_url:
        urls.extend([u.strip() for u in image_url.split(",") if u.strip()])

    if not urls:
        save_task(request_id, "glm", TaskStatus.failed, created_at)
        raise HTTPException(status_code=400, detail="没有可处理的图片 URL")

    results_data = await process_batch(urls, prompt, model)
    
    # 保存描述到OSS
    save_descriptions_to_oss(results_data, face_request_id, bucket)

    results = [
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

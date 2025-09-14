"""
数据模型定义
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class FaceDetectionAlgorithm(str, Enum):
    yolov8 = "yolov8"
    mtcnn = "mtcnn"
    blazeface = "blazeface"
    ultraface = "ultraface"
    opencv_dnn = "opencv_dnn"
    haar_cascade = "haar_cascade"


class ReturnType(str, Enum):
    url = "url"


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


class GLMBatchResponse(BaseModel):
    request_id: str
    status: TaskStatus
    created_at: float
    finished_at: float
    duration: float
    model: str
    total: int
    results: List[GLMItemResult]


# 更新前向引用
# 在新版本中不需要显式调用 model_rebuild()
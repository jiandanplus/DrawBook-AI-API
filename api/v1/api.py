import os
import io
import uuid
import time
import json
import logging
import shutil
import re
from typing import List, Optional
from enum import Enum
from fastapi import FastAPI, UploadFile, File, Form, APIRouter, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from PIL import Image
import oss2

from .models import (
    FaceDetectionAlgorithm,
    ReturnType,
    MultiFaceResponse,
    StatusResponse,
    GLMStatusResponse,
    GLMItemResult,
    GLMBatchResponse,
    TaskStatus,
    TrainingDataResponse,
    TrainingStatusResponse
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

# 新增接口：获取人脸处理结果并存储到本地训练目录
class ModelType(str, Enum):
    flux = "flux"
    sdxl = "sdxl"

@router_v1.post("/training_data", operation_id="prepare_training_data", response_model=TrainingDataResponse)
async def training_data(
    trigger: str = Form(..., description="触发词标识"),
    get_face_request_id: str = Form(..., description="人脸处理任务的request_id"),
    model_type: ModelType = Form(ModelType.flux, description="模型类型：flux 或 sdxl")
):
    """
    根据人脸处理任务ID，从OSS获取处理后的人脸图片和对应的描述文件，
    并存储到本地train_database/{trigger}目录下；同时将train.py中的model_type设置为该值
    """
    # 生成唯一的请求ID
    request_id = str(uuid.uuid4())
    created_at = time.time()
    
    # 统一使用枚举值
    model_type_value = model_type.value

    try:
        # 创建本地训练目录，结构为 train_database/{trigger}/1_{trigger}
        parent_dir = os.path.join("train_database", f"{trigger}")
        local_train_dir = os.path.join(parent_dir, f"1_{trigger}")
        os.makedirs(local_train_dir, exist_ok=True)

        # 1. 获取裁剪后的人脸图片
        corp_face_prefix = f"input_face/{get_face_request_id}/corp_face/"
        logger.info(f"开始获取裁剪后的人脸图片，前缀: {corp_face_prefix}")

        # 遍历OSS中指定前缀的所有文件
        corp_face_objects = []
        for obj in oss2.ObjectIterator(bucket, prefix=corp_face_prefix):
            if not obj.key.endswith('/'):
                corp_face_objects.append(obj.key)

        # 下载裁剪后的人脸图片到本地训练目录
        for obj_key in corp_face_objects:
            file_name = os.path.basename(obj_key)
            local_file_path = os.path.join(local_train_dir, file_name)
            bucket.get_object_to_file(obj_key, local_file_path)
            logger.info(f"已下载裁剪人脸图片: {obj_key} -> {local_file_path}")

        # 2. 获取人脸图片对应的描述文件
        prompt_prefix = f"input_face/{get_face_request_id}/per_face&prompt/"
        logger.info(f"开始获取描述文件，前缀: {prompt_prefix}")

        # 遍历OSS中指定前缀的所有文件
        prompt_objects = []
        for obj in oss2.ObjectIterator(bucket, prefix=prompt_prefix):
            if not obj.key.endswith('/'):
                prompt_objects.append(obj.key)

        # 下载描述文件到本地训练目录
        for obj_key in prompt_objects:
            file_name = os.path.basename(obj_key)
            local_file_path = os.path.join(local_train_dir, file_name)
            bucket.get_object_to_file(obj_key, local_file_path)
            logger.info(f"已下载描述文件: {obj_key} -> {local_file_path}")

        # 根据 model_type 更新 train_by_toml.py 中的 flux 标志 (flux -> 1, sdxl -> 0)
        try:
            train_by_toml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "train_by_toml.py"))
            if os.path.exists(train_by_toml_path):
                with open(train_by_toml_path, "r", encoding="utf-8") as f:
                    tb_content = f.read()

                new_flux = "1" if model_type_value == ModelType.flux.value else "0"

                def _replace_flux(m):
                    return m.group(1) + new_flux

                new_tb_content, n_tb = re.subn(r'^(\s*flux\s*=\s*)(\d+)', _replace_flux, tb_content, count=1, flags=re.M)
                if n_tb > 0:
                    with open(train_by_toml_path, "w", encoding="utf-8") as f:
                        f.write(new_tb_content)
                    logger.info(f"已将 {train_by_toml_path} 中的 flux 设置为 {new_flux}")
                else:
                    logger.warning(f"未在 {train_by_toml_path} 中找到 flux 赋值行，未修改文件")
            else:
                logger.warning(f"train_by_toml.py 未找到：{train_by_toml_path}")
        except Exception as e:
            logger.error(f"更新 train_by_toml.py 时出错: {e}")
            # 不影响主流程，仅记录错误

        # 根据 model_type 和 trigger 更新对应 toml 文件中的 output_name 和 train_data_dir
        try:
            toml_filename = "flux.toml" if model_type_value == ModelType.flux.value else "lork.toml"
            toml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "toml", toml_filename))
            if os.path.exists(toml_path):
                with open(toml_path, "r", encoding="utf-8") as f:
                    toml_content = f.read()

                def _replace_output_name(m):
                    return m.group(1) + f'"{trigger}"'

                def _replace_train_data_dir(m):
                    return m.group(1) + f'"./train_database/{trigger}"'

                # 更新 output_name
                new_toml_content, n_output = re.subn(r'^(\s*output_name\s*=\s*).*$', _replace_output_name, toml_content, count=1, flags=re.M)
                
                # 更新 train_data_dir
                new_toml_content, n_train_dir = re.subn(r'^(\s*train_data_dir\s*=\s*).*$', _replace_train_data_dir, new_toml_content, count=1, flags=re.M)
                
                if n_output > 0 or n_train_dir > 0:
                    with open(toml_path, "w", encoding="utf-8") as f:
                        f.write(new_toml_content)
                    
                    updates = []
                    if n_output > 0:
                        updates.append(f"output_name='{trigger}'")
                    if n_train_dir > 0:
                        updates.append(f"train_data_dir='./train_database/{trigger}'")
                    
                    logger.info(f"已将 {toml_path} 中的 {', '.join(updates)} 更新")
                else:
                    logger.warning(f"未在 {toml_path} 中找到 output_name 或 train_data_dir 赋值行，未修改文件")
            else:
                logger.warning(f"TOML 文件未找到：{toml_path}")
        except Exception as e:
            logger.error(f"更新 TOML 文件时出错: {e}")
            # 不影响主流程，仅记录错误

        # 启动训练进程
        training_started = False
        training_pid = None
        log_file_path = None
        try:
            train_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "train_by_toml.py"))
            if os.path.exists(train_script_path):
                # 创建训练日志目录和文件
                log_dir = os.path.join("logs", "training")
                os.makedirs(log_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                log_file_path = os.path.join(log_dir, f"training_{trigger}_{timestamp}.log")
                
                # 修复：使用与手动运行相同的方式启动进程
                import subprocess
                import sys
                
                # 确保使用相同的工作目录和环境
                work_dir = os.path.dirname(train_script_path)
                
                # 使用更稳定的进程启动方式
                if os.name == 'nt':  # Windows
                    # 在 Windows 上使用 CREATE_NEW_PROCESS_GROUP 而不是 DETACHED_PROCESS
                    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    creation_flags = 0
                
                # 打开日志文件
                log_file_handle = open(log_file_path, "w", encoding="utf-8")
                
                # 启动进程，使用与手动运行相同的命令
                process = subprocess.Popen(
                    [sys.executable, train_script_path],
                    stdout=log_file_handle,
                    stderr=subprocess.STDOUT,
                    cwd=work_dir,  # 确保工作目录正确
                    creationflags=creation_flags,
                    # 继承父进程的环境变量
                    env=os.environ.copy()
                )
                
                training_pid = process.pid
                training_started = True
                logger.info(f"已启动训练进程，PID: {training_pid}, 日志文件: {log_file_path}")
                
                # 短暂等待确保进程启动成功
                time.sleep(2)
                
                # 检查进程是否还在运行（验证启动成功）
                poll_result = process.poll()
                if poll_result is not None:
                    # 进程已经退出，读取错误信息
                    log_file_handle.close()
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        error_content = f.read()
                    logger.error(f"训练进程启动失败，退出代码: {poll_result}")
                    logger.error(f"错误日志: {error_content}")
                    training_started = False
                    training_pid = None
                else:
                    logger.info(f"训练进程 {training_pid} 启动成功并正在运行")
                
                # 将训练信息保存到任务数据库
                training_task_data = {
                    "pid": training_pid,
                    "trigger": trigger,
                    "model_type": model_type_value,
                    "log_file": log_file_path,
                    "started_at": time.time(),
                    "log_file_handle": log_file_handle  # 保持句柄开放
                }
                save_task(f"training_{training_pid}", "training", TaskStatus.running, 
                         time.time(), None, None, results=training_task_data)
            else:
                logger.warning(f"train_by_toml.py 未找到：{train_script_path}")
        except Exception as e:
            logger.error(f"启动训练进程时出错: {e}")
            # 不影响主流程，仅记录错误

        # 计算完成时间和持续时间
        finished_at = time.time()
        duration = finished_at - created_at

        # 构造完整的响应数据
        response_data = TrainingDataResponse(
            request_id=request_id,
            status="success",
            message=f"训练数据准备完成，共下载 {len(corp_face_objects)} 张人脸图片和 {len(prompt_objects)} 个描述文件",
            trigger=trigger,
            get_face_request_id=get_face_request_id,
            model_type=model_type_value,
            corp_face_count=len(corp_face_objects),
            prompt_file_count=len(prompt_objects),
            local_train_dir=local_train_dir,
            training_started=training_started,
            training_pid=training_pid,
            log_file=log_file_path,
            created_at=created_at,
            finished_at=finished_at,
            duration=duration
        )

        # 保存完整的响应数据到数据库
        save_task(request_id, "prepare_training", TaskStatus.completed, created_at, finished_at, duration, results=response_data)

        # 返回成功响应
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"准备训练数据时出错: {e}")
        raise HTTPException(status_code=500, detail=f"准备训练数据时出错: {str(e)}")


@router_v1.get("/training_status/{task_id}", operation_id="get_training_status", response_model=TrainingStatusResponse)
def get_training_status(task_id: str):
    """
    根据任务ID查询训练状态，通过解析日志文件来获取详细进度。
    支持 prepare_training 的 request_id 或 training_<pid> 格式的 ID。
    """
    # 1. 在数据库中查找任务记录
    task = get_task(task_id)
    training_task = None
    
    if task:
        if task.type == "prepare_training":
            # 如果是主任务，从中提取训练任务信息
            try:
                results = json.loads(task.results) if task.results else {}
                pid = results.get("training_pid")
                if pid:
                    training_task = get_task(f"training_{pid}")
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(status_code=404, detail="主任务结果格式错误，无法找到训练任务")
        elif task.type == "training":
            training_task = task

    if not training_task:
        # 如果通过主任务ID找不到，尝试直接把 task_id 当作 training_pid 去构造
        if 'training_' not in task_id:
             # 尝试通过 pid 直接查找
             potential_training_id = f"training_{task_id}"
             training_task = get_task(potential_training_id)

    if not training_task:
        raise HTTPException(status_code=404, detail=f"未找到与 {task_id} 相关的训练任务")

    # 2. 从训练任务记录中解析 PID 和日志文件路径
    try:
        train_results = json.loads(training_task.results) if training_task.results else {}
        pid = train_results.get("pid")
        log_file = train_results.get("log_file")
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=500, detail="训练任务记录已损坏")

    if not pid or not log_file:
        raise HTTPException(status_code=404, detail="训练任务记录不完整，缺少PID或日志文件路径")

    # 3. 检查进程是否仍在运行
    is_running = False
    try:
        import psutil
        if pid:
            is_running = psutil.pid_exists(pid)
    except ImportError:
        logger.warning("psutil 未安装，无法准确判断进程状态")
    except Exception as e:
        logger.error(f"检查进程状态时出错: {e}")

    # 4. 解析日志文件以确定详细状态和进度
    status = "creating"
    progress = 0.0
    log_preview = []

    if not os.path.exists(log_file):
        # 日志文件还未创建，任务处于非常早期的创建阶段
        return TrainingStatusResponse(
            task_id=task_id,
            status=status,
            progress=progress,
            is_process_running=is_running
        )

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        log_preview = [line.strip() for line in lines[-10:]] # 获取最后10行作为预览

        log_content = "".join(lines)

        if "Training finished successfully" in log_content:
            status = "completed"
            progress = 100.0
        elif "running training / 学習開始" in log_content:
            status = "running"
            # 从日志中匹配进度
            # steps: 100%|██████████| 1600/1600 [01:23<00:00, 19.09it/s, loss=0.05, lr=1e-6]
            progress_matches = re.findall(r"steps:\s+(\d+)%", log_content)
            if progress_matches:
                progress = float(progress_matches[-1]) # 取最后一个匹配到的进度
        else:
            # 如果没有明确的开始或结束标志，则认为是 "creating"
            status = "creating"

        # 如果进程已经不在了，但日志没有显示成功，则标记为失败
        if not is_running and status != "completed":
            # 检查是否有错误信息
            if any("error" in line.lower() or "traceback" in line.lower() for line in lines):
                 status = "failed"
            else:
                 # 可能是未知原因的提前终止
                 status = "terminated"

    except Exception as e:
        logger.error(f"解析日志文件 {log_file} 时出错: {e}")
        # 即使日志解析失败，也返回基本信息
        return TrainingStatusResponse(
            task_id=task_id,
            status="log_error",
            progress=0.0,
            is_process_running=is_running,
            log_preview=[f"Error reading log file: {e}"]
        )

    return TrainingStatusResponse(
        task_id=task_id,
        status=status,
        progress=progress,
        is_process_running=is_running,
        log_preview=log_preview
    )


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
    elif task.type == "prepare_training":
        # 训练准备任务使用TrainingDataResponse模型
        return TrainingDataResponse(**results)
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

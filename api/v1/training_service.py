import os
import time
import uuid
import json
import logging
import re
from typing import Optional

import oss2

from .models import TrainingDataResponse, TrainingStatusResponse, TaskStatus
from .database import save_task, get_task, list_tasks
from .oss_utils import bucket

logger = logging.getLogger(__name__)


def prepare_training_data_service(trigger: str, get_face_request_id: str, model_type_value: str) -> TrainingDataResponse:
    request_id = str(uuid.uuid4())
    created_at = time.time()

    # 创建本地训练目录
    parent_dir = os.path.join("train_database", f"{trigger}")
    local_train_dir = os.path.join(parent_dir, f"1_{trigger}")
    os.makedirs(local_train_dir, exist_ok=True)

    # 下载裁剪后的人脸图片
    corp_face_prefix = f"input_face/{get_face_request_id}/corp_face/"
    logger.info(f"开始获取裁剪后的人脸图片，前缀: {corp_face_prefix}")
    corp_face_objects = []
    for obj in oss2.ObjectIterator(bucket, prefix=corp_face_prefix):
        if not obj.key.endswith('/'):
            corp_face_objects.append(obj.key)

    for obj_key in corp_face_objects:
        file_name = os.path.basename(obj_key)
        local_file_path = os.path.join(local_train_dir, file_name)
        bucket.get_object_to_file(obj_key, local_file_path)
        logger.info(f"已下载裁剪人脸图片: {obj_key} -> {local_file_path}")

    # 下载描述文件
    prompt_prefix = f"input_face/{get_face_request_id}/per_face&prompt/"
    logger.info(f"开始获取描述文件，前缀: {prompt_prefix}")
    prompt_objects = []
    for obj in oss2.ObjectIterator(bucket, prefix=prompt_prefix):
        if not obj.key.endswith('/'):
            prompt_objects.append(obj.key)

    for obj_key in prompt_objects:
        file_name = os.path.basename(obj_key)
        local_file_path = os.path.join(local_train_dir, file_name)
        bucket.get_object_to_file(obj_key, local_file_path)
        logger.info(f"已下载描述文件: {obj_key} -> {local_file_path}")

    # 更新 train_by_toml.py 中 flux
    try:
        train_by_toml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train_by_toml.py'))
        if os.path.exists(train_by_toml_path):
            with open(train_by_toml_path, 'r', encoding='utf-8') as f:
                tb_content = f.read()
            new_flux = '1' if model_type_value == 'flux' else '0'

            def _replace_flux(m):
                return m.group(1) + new_flux

            new_tb_content, n_tb = re.subn(r'^(\s*flux\s*=\s*)(\d+)', _replace_flux, tb_content, count=1, flags=re.M)
            if n_tb > 0:
                with open(train_by_toml_path, 'w', encoding='utf-8') as f:
                    f.write(new_tb_content)
                logger.info(f"已将 {train_by_toml_path} 中的 flux 设置为 {new_flux}")
    except Exception as e:
        logger.error(f"更新 train_by_toml.py 时出错: {e}")

    # 更新 toml 文件
    try:
        toml_filename = 'flux.toml' if model_type_value == 'flux' else 'lork.toml'
        toml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'toml', toml_filename))
        if os.path.exists(toml_path):
            with open(toml_path, 'r', encoding='utf-8') as f:
                toml_content = f.read()

            def _replace_output_name(m):
                return m.group(1) + f'"{trigger}"'

            def _replace_train_data_dir(m):
                return m.group(1) + f'"./train_database/{trigger}"'

            def _replace_output_dir(m):
                return m.group(1) + f'"./output/{trigger}"'

            new_toml_content, n_output = re.subn(r'^(\s*output_name\s*=\s*).*$', _replace_output_name, toml_content, count=1, flags=re.M)
            new_toml_content, n_train_dir = re.subn(r'^(\s*train_data_dir\s*=\s*).*$', _replace_train_data_dir, new_toml_content, count=1, flags=re.M)
            new_toml_content, n_output_dir = re.subn(r'^(\s*output_dir\s*=\s*).*$', _replace_output_dir, new_toml_content, count=1, flags=re.M)
            if n_output > 0 or n_train_dir > 0 or n_output_dir > 0:
                with open(toml_path, 'w', encoding='utf-8') as f:
                    f.write(new_toml_content)
                updates = []
                if n_output > 0:
                    updates.append(f"output_name='{trigger}'")
                if n_train_dir > 0:
                    updates.append(f"train_data_dir='./train_database/{trigger}'")
                if n_output_dir > 0:
                    updates.append(f"output_dir='./output/{trigger}'")
                logger.info(f"已将 {toml_path} 中的 {', '.join(updates)} 更新")
    except Exception as e:
        logger.error(f"更新 TOML 文件时出错: {e}")

    # 保存最小主任务记录
    try:
        save_task(request_id, 'prepare_training', TaskStatus.pending, created_at, None, None,
                  results={
                      'trigger': trigger,
                      'get_face_request_id': get_face_request_id,
                      'model_type': model_type_value,
                      'message': 'creating'
                  })
    except Exception as e:
        logger.error(f"保存初始主任务记录时出错: {e}")

    # 启动训练脚本
    training_started = False
    training_pid: Optional[int] = None
    log_file_path: Optional[str] = None
    try:
        train_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'train_by_toml.py'))
        if os.path.exists(train_script_path):
            log_dir = os.path.join('logs', 'training')
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            log_file_path = os.path.join(log_dir, f'training_{trigger}_{timestamp}.log')

            import subprocess
            import sys

            work_dir = os.path.dirname(train_script_path)
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0

            log_file_handle = open(log_file_path, 'w', encoding='utf-8')
            process = subprocess.Popen([
                sys.executable, train_script_path
            ], stdout=log_file_handle, stderr=subprocess.STDOUT, cwd=work_dir, creationflags=creation_flags, env=os.environ.copy())

            training_pid = process.pid
            training_started = True
            logger.info(f"已启动训练进程，PID: {training_pid}, 日志文件: {log_file_path}")

            time.sleep(2)
            poll_result = process.poll()
            if poll_result is not None:
                log_file_handle.close()
                with open(log_file_path, 'r', encoding='utf-8') as f:
                    error_content = f.read()
                logger.error(f"训练进程启动失败，退出代码: {poll_result}")
                logger.error(f"错误日志: {error_content}")
                training_started = False
                training_pid = None
            else:
                logger.info(f"训练进程 {training_pid} 启动成功并正在运行")

            training_task_data = {
                'pid': training_pid,
                'trigger': trigger,
                'model_type': model_type_value,
                'log_file': log_file_path,
                'started_at': time.time(),
            }
            try:
                save_task(f"training_{training_pid}", 'training', TaskStatus.running, time.time(), None, None, results=training_task_data)
            except Exception as e:
                logger.error(f"保存训练子任务到数据库时出错: {e}")

            try:
                main_results = {
                    'trigger': trigger,
                    'get_face_request_id': get_face_request_id,
                    'model_type': model_type_value,
                    'training_started': training_started,
                    'training_pid': training_pid,
                    'log_file': log_file_path,
                    'corp_face_count': len(corp_face_objects),
                    'prompt_file_count': len(prompt_objects),
                    'local_train_dir': local_train_dir,
                    'message': 'training_started' if training_started else 'training_not_started'
                }
                save_task(request_id, 'prepare_training', TaskStatus.running, created_at, None, None, results=main_results)
            except Exception as e:
                logger.error(f"更新主任务记录时出错: {e}")

    except Exception as e:
        logger.error(f"启动训练进程时出错: {e}")
        try:
            finished_at_err = time.time()
            save_task(request_id, 'prepare_training', TaskStatus.failed, created_at, finished_at_err, finished_at_err - created_at,
                      results={'error': str(e)})
        except Exception as e2:
            logger.error(f"在处理训练启动异常时更新主任务失败: {e2}")

    finished_at = time.time()
    duration = finished_at - created_at

    response_data = TrainingDataResponse(
        request_id=request_id,
        status='success',
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

    try:
        save_task(request_id, 'prepare_training', TaskStatus.completed, created_at, finished_at, duration, results=response_data)
    except Exception as e:
        logger.error(f"保存完整主任务记录失败: {e}")

    return response_data


def get_training_status_service(task_id: str) -> TrainingStatusResponse:
    task = get_task(task_id)
    training_task = None
    if task:
        if task.type == 'prepare_training':
            try:
                results = json.loads(task.results) if task.results else {}
                pid = results.get('training_pid')
                if pid:
                    training_task = get_task(f"training_{pid}")
            except Exception:
                raise
        elif task.type == 'training':
            training_task = task

    if not training_task:
        if 'training_' not in task_id:
            potential_training_id = f"training_{task_id}"
            training_task = get_task(potential_training_id)

    if not training_task:
        raise RuntimeError(f"未找到与 {task_id} 相关的训练任务")

    try:
        train_results = json.loads(training_task.results) if training_task.results else {}
        pid = train_results.get('pid')
        log_file = train_results.get('log_file')
    except Exception:
        raise

    is_running = False
    try:
        import psutil
        if pid:
            is_running = psutil.pid_exists(pid)
    except Exception:
        logger.warning('psutil 未安装或检查失败')

    status = 'creating'
    progress = 0.0
    log_preview = []

    if not log_file or not os.path.exists(log_file):
        return TrainingStatusResponse(task_id=task_id, status=status, progress=progress, is_process_running=is_running)

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        log_preview = [line.strip() for line in lines[-10:]]
        log_content = ''.join(lines)
        if 'Training finished successfully' in log_content:
            status = 'completed'
            progress = 100.0
        elif 'running training / 学習開始' in log_content:
            status = 'running'
            progress_matches = re.findall(r'steps:\s+(\d+)%', log_content)
            if progress_matches:
                progress = float(progress_matches[-1])
        else:
            status = 'creating'

        if not is_running and status != 'completed':
            if any('error' in line.lower() or 'traceback' in line.lower() for line in lines):
                status = 'failed'
            else:
                status = 'terminated'
    except Exception as e:
        logger.error(f"解析日志文件 {log_file} 时出错: {e}")
        return TrainingStatusResponse(task_id=task_id, status='log_error', progress=0.0, is_process_running=is_running,
                                      log_preview=[f"Error reading log file: {e}"])

    return TrainingStatusResponse(task_id=task_id, status=status, progress=progress, is_process_running=is_running, log_preview=log_preview)


def stop_training_service(task_id: str):
    task = get_task(task_id)
    training_task = None
    if task:
        if task.type == 'prepare_training':
            try:
                results = json.loads(task.results) if task.results else {}
                pid = results.get('training_pid') or results.get('pid')
                if pid:
                    training_task = get_task(f"training_{pid}")
            except Exception:
                pass
        elif task.type == 'training':
            training_task = task

    if not training_task:
        potential = task_id
        if not potential.startswith('training_'):
            potential = f"training_{task_id}"
        training_task = get_task(potential)

    if not training_task:
        raise RuntimeError(f"未找到训练任务: {task_id}")

    try:
        train_results = json.loads(training_task.results) if training_task.results else {}
        pid = int(train_results.get('pid')) if train_results.get('pid') else None
    except Exception:
        pid = None

    if not pid:
        raise RuntimeError('训练任务记录中未包含 PID，无法停止')

    stopped = False
    info_msgs = []
    try:
        import psutil
        try:
            p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            info_msgs.append(f"进程 {pid} 不存在")
            stopped = True
        else:
            children = p.children(recursive=True)
            for c in children:
                try:
                    c.terminate()
                    info_msgs.append(f"终止子进程 {c.pid}")
                except Exception as e:
                    info_msgs.append(f"终止子进程 {c.pid} 失败: {e}")
            try:
                p.terminate()
                info_msgs.append(f"终止主进程 {pid}")
            except Exception as e:
                info_msgs.append(f"terminate 主进程失败: {e}")

            gone, alive = psutil.wait_procs([p] + children, timeout=5)
            if alive:
                for proc in alive:
                    try:
                        proc.kill()
                        info_msgs.append(f"强制杀死进程 {proc.pid}")
                    except Exception as e:
                        info_msgs.append(f"强制杀死进程 {proc.pid} 失败: {e}")
            stopped = True
    except ImportError:
        info_msgs.append('psutil 未安装，跳过优雅终止')

    if not stopped:
        try:
            if os.name == 'nt':
                cmd = f"taskkill /PID {pid} /T /F"
                rc = os.system(cmd)
                info_msgs.append(f"执行: {cmd}, rc={rc}")
                stopped = rc == 0
            else:
                try:
                    os.kill(pid, 9)
                    info_msgs.append(f"发送 SIGKILL 到 {pid}")
                    stopped = True
                except Exception as e:
                    info_msgs.append(f"无法杀死进程 {pid}: {e}")
        except Exception as e:
            info_msgs.append(f"使用 taskkill/os.kill 时出错: {e}")

    finished_at = time.time()
    duration = finished_at - (getattr(training_task, 'created_at', finished_at) or finished_at)
    try:
        save_task(training_task.request_id, 'training', TaskStatus.failed, training_task.created_at, finished_at, duration,
                  results={'pid': pid, 'stopped': stopped, 'info': info_msgs})
    except Exception as e:
        logger.error(f"更新训练任务状态失败: {e}")

    try:
        candidates = list_tasks(1, 50)
        for c in candidates:
            if c.type == 'prepare_training' and c.results:
                try:
                    r = json.loads(c.results)
                    if r.get('training_pid') == pid or r.get('training_pid') == str(pid):
                        save_task(c.request_id, 'prepare_training', TaskStatus.failed, c.created_at, finished_at, finished_at - c.created_at,
                                  results={'training_pid': pid, 'stopped': stopped, 'info': info_msgs})
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"尝试更新主任务状态时出错: {e}")

    return {'pid': pid, 'stopped': stopped, 'info': info_msgs}

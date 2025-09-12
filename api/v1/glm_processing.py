"""
GLM处理相关功能
"""
import os
import re
import json
import time
import asyncio
import logging
import concurrent.futures
from typing import List, Optional
from urllib.parse import urlparse
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from .models import TaskStatus, GLMItemResult
from .database import get_task
from .oss_utils import save_description_to_oss

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)

# ----------------- ZhipuAI 工具 -----------------
client = None


def init_zhipu_client(api_key: str = None) -> bool:
    global client
    try:
        # 优先使用传入的api_key，其次从环境变量获取
        key = api_key or os.getenv("GLM_API_KEY")
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


def get_image_urls_from_face_task(face_request_id: str) -> List[str]:
    """
    从人脸处理任务中获取图片URL列表
    """
    urls: List[str] = []
    
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
    
    return urls


def save_descriptions_to_oss(results_data: List[dict], face_request_id: str, bucket):
    """
    保存描述到OSS
    """
    if face_request_id:
        for result in results_data:
            if isinstance(result, dict) and result.get("status") == TaskStatus.completed and result.get("description"):
                try:
                    save_description_to_oss(result["description"], result["image_url"], face_request_id, bucket)
                except Exception as e:
                    logger.error(f"保存描述文件时出错: {e}")
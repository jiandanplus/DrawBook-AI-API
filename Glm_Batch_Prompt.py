import os
import re
import json
import time
import asyncio
import requests
from typing import List
from zhipuai import ZhipuAI
import concurrent.futures
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

def init_zhipu_client(api_key: str = None):
    """
    初始化ZhipuAI客户端
    """
    global client
    try:
        # 优先使用传入的api_key
        if api_key:
            client = ZhipuAI(api_key=api_key)
            return True
        
        # 从环境变量中获取API Key
        env_api_key = os.getenv("ZHIPUAI_API_KEY")
        if env_api_key:
            client = ZhipuAI(api_key=env_api_key)
            return True
            
        # 尝试使用其他方式获取API Key
        client = ZhipuAI()
        return True
        
    except Exception as e:
        print(f"警告：无法初始化ZhipuAI客户端: {e}")
        client = None
        return False

def download_image(image_url: str) -> str:
    """
    下载图片到output目录
    """
    try:
        # 解析URL获取文件名
        parsed_url = urlparse(image_url)
        base_name = os.path.basename(parsed_url.path)
        
        # 如果没有文件名或扩展名，使用默认名称
        if not base_name or '.' not in base_name:
            base_name = 'image.jpg'  # 默认文件名
        
        # 清理文件名
        file_name = sanitize_filename(base_name)
        
        # 创建输出目录（如果不存在）
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建文件路径
        file_path = os.path.join(output_dir, file_name)
        
        # 如果文件已存在，添加序号
        counter = 1
        original_file_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_file_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1
        
        # 下载图片
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # 保存图片
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"图片已下载到: {file_path}")
        return file_path
    except Exception as e:
        print(f"下载图片 {image_url} 时出错: {e}")
        return None
    
def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不合法的字符
    """
    # 边界条件检查
    if not filename:
        return filename
    
    # 使用正则表达式一次性替换所有非法字符
    # 包含Windows和Unix系统下的非法字符
    illegal_chars_pattern = r'[<>:"/\\|?*\x00-\x1f]'
    filename = re.sub(illegal_chars_pattern, '_', filename)
    
    return filename

def save_description_to_file(image_url: str, description: str, prompt: str):
    """
    将描述内容保存为txt文件
    """
    # 解析URL获取文件名
    parsed_url = urlparse(image_url)
    base_name = os.path.basename(parsed_url.path)
    
    # 移除文件扩展名
    if '.' in base_name:
        file_name = '.'.join(base_name.split('.')[:-1])
    else:
        file_name = base_name
    
    # 如果文件名为空，则使用URL的host部分
    if not file_name:
        file_name = parsed_url.netloc.replace('.', '_')
    
    # 清理文件名
    file_name = sanitize_filename(file_name)
    
    # 创建输出目录（如果不存在）
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建文件路径
    file_path = os.path.join(output_dir, f"{file_name}.txt")
    
    # 处理描述中的标签
    cleaned_description = description
    if "<|begin_of_box|>" in cleaned_description and "<|end_of_box|>" in cleaned_description:
        # 移除标签
        cleaned_description = cleaned_description.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    
    # 写入文件
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # f.write(f"Image URL: {image_url}\n")
            # f.write(f"Prompt: {prompt}\n")
            # f.write("=" * 50 + "\n")
            f.write(cleaned_description)
        print(f"描述已保存到: {file_path}")
    except Exception as e:
        print(f"保存文件 {file_path} 时出错: {e}")

# 修改 fetch_image_description_sync 函数，接收并使用 model 参数
def fetch_image_description_sync(image_url: str, prompt: str, model: str = "glm-4.5v") -> dict:
    """
    使用ZhipuAI GLM-4V模型获取图片描述（同步版本）
    """
    if not client:
        return {
            "image_url": image_url,
            "prompt": prompt,
            "error": "ZhipuAI客户端未初始化",
            "status": "error"
        }
    
    try:
        # 下载图片到本地
        image_path = download_image(image_url)
        
        # 调用GLM-4V API，使用传入的 model 参数
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.5,
            max_tokens=4096,
        )
        
        description = response.choices[0].message.content
        result = {
            "image_url": image_url,
            "prompt": prompt,
            "description": description,
            "image_path": image_path,  # 添加图片路径到返回结果
            "status": "success"
        }
        print(f"完成处理: {image_url}")
        
        # 保存描述到文件
        save_description_to_file(image_url, description, prompt)
        
        return result
    except Exception as e:
        return {
            "image_url": image_url,
            "prompt": prompt,
            "error": str(e),
            "status": "error"
        }
    
# 修改 process_batch 函数，传递 model 参数
async def process_batch(batch: List[str], prompt: str, model: str) -> List[dict]:
    """
    处理一批图像请求
    """
    loop = asyncio.get_event_loop()
    tasks = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for image_url in batch:
            # 将同步调用放到线程池中执行，传递 model 参数
            task = loop.run_in_executor(executor, fetch_image_description_sync, image_url, prompt, model)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# 在 main 函数中读取配置的地方添加 model 配置读取
async def main():
    # 记录开始时间
    start_time = time.time()
    print(f"任务开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # 读取配置文件
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    image_urls = config.get('image_urls', [])
    prompt = config.get('prompt', '')
    concurrent_requests = config.get('concurrent_requests', len(image_urls))
    model = config.get('model', 'glm-4.5v')  # 添加这一行，默认值为 'glm-4.5v'
    
    # 初始化ZhipuAI客户端（不从config.json中读取API Key，而是使用环境变量）
    init_zhipu_client()
    print(f"总共需要处理 {len(image_urls)} 个图片")
    print(f"并发请求数量: {concurrent_requests}")
    
    if not client:
        print("错误：ZhipuAI客户端未正确初始化，请检查API Key配置")
        return
    
    # 如果请求数量大于batch_size，则分批处理，每批batch_size个任务
    batch_size = 100
    if concurrent_requests > batch_size:
        # 分批处理
        for i in range(0, len(image_urls), batch_size):
            batch = image_urls[i:i+batch_size]
            print(f"正在处理批次 {i//batch_size + 1}: {len(batch)} 个任务")
            
            batch_results = await process_batch(batch, prompt,model)
            
            # 处理当前批次结果
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"处理出错: {result}")
                elif isinstance(result, dict) and result.get("status") == "error":
                    print(f"请求出错: {result.get('error')}")
                else:
                    print(f"成功获取描述: {result.get('image_url') if isinstance(result, dict) else '未知'}")
                    if isinstance(result, dict) and 'description' in result:
                        print(f"描述内容: {result['description'][:100]}...")
            
            # 在批次之间添加间隔（如果需要）
            if i + batch_size < len(image_urls):
                await asyncio.sleep(0.5)
    else:
        # 直接并发处理所有请求
        print(f"直接处理所有 {len(image_urls)} 个请求")
        results = await process_batch(image_urls, prompt,model)
        
        # 处理结果
        for result in results:
            if isinstance(result, Exception):
                print(f"处理出错: {result}")
            elif isinstance(result, dict) and result.get("status") == "error":
                print(f"请求出错: {result.get('error')}")
            else:
                print(f"成功获取描述: {result.get('image_url') if isinstance(result, dict) else '未知'}")
                if isinstance(result, dict) and 'description' in result:
                    print(f"描述内容: {result['description'][:100]}...")
    
    # 记录结束时间
    end_time = time.time()
    print(f"任务结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    asyncio.run(main())
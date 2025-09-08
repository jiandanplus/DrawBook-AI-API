import uvicorn
from api.v1.api import app, cleanup_expired_tasks  # 导入你在 api/v1/api.py 里定义的 FastAPI app
import threading, time

def cleanup_loop():
    while True:
        cleanup_expired_tasks(expire_seconds=9999 * 24 * 3600)
        time.sleep(3600)  # 每小时清理一次

threading.Thread(target=cleanup_loop, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(
        "api.v1.api:app",   # 指定模块路径和 app 对象
        host="0.0.0.0",     # 对外暴露
        port=8000,          # 服务端口
        reload=True         # 开发时热重载
    )

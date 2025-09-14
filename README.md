# DrawBook AI API

脸模训练相关API接口

## 项目简介

本项目提供了一系列用于人脸处理和AI图像描述生成的API接口。

## 功能特性

1. 人脸裁剪和处理
2. AI图像描述生成（基于GLM模型）
3. 任务状态查询和管理
4. 阿里云OSS集成
5. MySQL数据库存储

## 环境要求

- Python 3.8+
- MySQL数据库
- 阿里云OSS账户
- GLM API密钥（可选，用于AI图像描述功能）

## 安装依赖

### 生产环境依赖
```bash
pip install -r requirements.txt
```

### 开发环境依赖
```bash
pip install -r requirements-dev.txt
```

## 环境配置

1. 复制 `.env.example` 文件为 `.env`：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，填入实际的配置信息：
   - 阿里云OSS访问密钥
   - GLM API密钥
   - 数据库连接信息

## 启动服务

```bash
# 使用uvicorn启动
uvicorn api.v1.api:app --host 0.0.0.0 --port 8000

# 或者使用main.py启动（包含定时清理任务）
python main.py
```

## API文档

服务启动后，访问以下地址查看API文档：
- FastAPI自动生成文档：http://localhost:8000/api/v1/docs
- ReDoc文档：http://localhost:8000/api/v1/redoc

## 项目结构

```
.
├── api/                    # API接口代码
│   └── v1/                # v1版本API
│       ├── models/        # 数据模型
│       ├── api.py         # API路由和接口定义
│       ├── database.py    # 数据库相关代码
│       ├── face_processing.py  # 人脸处理功能
│       ├── glm_processing.py   # GLM处理功能
│       └── oss_utils.py   # OSS工具函数
├── models/                # AI模型文件
├── static/                # 静态文件目录
├── .env                   # 环境变量配置文件
├── .env.example           # 环境变量示例文件
├── requirements.txt       # 生产环境依赖
├── requirements-dev.txt   # 开发环境依赖
├── main.py               # 服务启动文件
└── crop_face.py          # 人脸裁剪核心功能
```

## 主要API接口

1. `POST /api/v1/get_faces` - 人脸裁剪接口
2. `POST /api/v1/glm_batch_prompt` - 批量图像描述生成
3. `GET /api/v1/get_task/{request_id}` - 查询任务状态
4. `GET /api/v1/list_tasks` - 分页查询任务列表

## 注意事项

1. 人脸检测支持多种算法（YOLOv8、MTCNN等），需要相应的模型文件
2. 需要配置阿里云OSS以存储处理后的图像
3. 图像描述生成功能需要有效的GLM API密钥
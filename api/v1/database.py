"""
数据库相关功能
"""
import json
import time
import os
from typing import List, Optional
from sqlalchemy import Column, String, Float, Text, Enum as SqlEnum, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from dotenv import load_dotenv
from .models import TaskStatus

# 加载环境变量
load_dotenv()

# ---------- SQLAlchemy 初始化 ----------
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


# MySQL 连接配置
DB_USER = os.getenv("DB_USER", "drawbook")
DB_PASSWORD = os.getenv("DB_PASSWORD", "WwY2fjcMCDyjDDxJ")
DB_HOST = os.getenv("DB_HOST", "139.196.51.51")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "drawbook")

MYSQL_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(MYSQL_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = scoped_session(sessionmaker(bind=engine, autocommit=False, autoflush=False))

# 创建表
Base.metadata.create_all(engine)


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
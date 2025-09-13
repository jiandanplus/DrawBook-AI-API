#!/usr/bin/env python3
"""
Script to remove files unrelated to flux_train_network.py
只保留与 flux_train_network.py 相关的文件
"""

import os
import shutil
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent

# 定义根目录需要保留的文件
ROOT_FILES = {
    "flux_train_network.py",
    "train_network.py",
    "requirements.txt",
    "setup.py",
    "README.md",
    "README-ja.md",
    "LICENSE.md",
    "_typos.toml",
    "pytest.ini"
}

# 定义需要保留的库文件
LIBRARY_FILES = {
    "__init__.py",
    "device_utils.py",
    "flux_models.py",
    "flux_train_utils.py",
    "flux_utils.py",
    "strategy_base.py",
    "strategy_flux.py",
    "train_util.py",
    "utils.py",
    "config_util.py",
    "custom_train_functions.py",
    "sd3_train_utils.py"
}

# 定义需要保留的测试文件
TEST_FILES = {
    "__init__.py",
    "test_flux_train_network.py",
    "library/test_flux_train_utils.py"
}

# 定义需要保留的工具文件
TOOLS_FILES = {
    "convert_diffusers_to_flux.py"
}

# 定义需要保留的网络文件
NETWORKS_FILES = {
    "__init__.py",
    "lora_flux.py",
    "oft_flux.py",
    "lora.py",
    "oft.py"
}

def main():
    print("This script will remove files unrelated to flux_train_network.py")
    print("Files/directories to be kept:")
    
    # 统计所有需要保留的文件
    keep_set = set()
    
    # 添加根目录文件
    for file in ROOT_FILES:
        keep_set.add(file)
        print(f"  {file}")
        
    # 添加库文件
    for file in LIBRARY_FILES:
        path = f"library/{file}"
        keep_set.add(path)
        print(f"  {path}")
        
    # 添加测试文件
    for file in TEST_FILES:
        path = f"tests/{file}"
        keep_set.add(path)
        print(f"  {path}")
        
    # 添加工具文件
    for file in TOOLS_FILES:
        path = f"tools/{file}"
        keep_set.add(path)
        print(f"  {path}")
        
    # 添加网络模块文件
    for file in NETWORKS_FILES:
        path = f"networks/{file}"
        keep_set.add(path)
        print(f"  {path}")
    
    # 添加需要保留的目录
    keep_dirs = {
        "library",
        "tests",
        "tools",
        "networks"
    }
    
    # 确认操作
    print(f"\nAbout to remove all files except the {len(keep_set)} files/directories listed above.")
    confirm = input("Are you sure you want to remove all other files? Type 'yes' to confirm: ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    # 执行删除操作
    deleted_count = 0
    
    # 遍历项目根目录
    for item in PROJECT_ROOT.iterdir():
        if item.name.startswith('.') or item.name == 'cleanup_unrelated_files.py':
            # 跳过隐藏文件和当前脚本
            continue
            
        if item.is_file() and item.name not in ROOT_FILES:
            try:
                item.unlink()
                print(f"Deleted file: {item.name}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting file {item.name}: {e}")
        elif item.is_dir() and item.name not in keep_dirs:
            try:
                shutil.rmtree(item)
                print(f"Deleted directory: {item.name}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting directory {item.name}: {e}")
    
    # 处理库目录中的文件
    library_path = PROJECT_ROOT / "library"
    if library_path.exists():
        for item in library_path.iterdir():
            if item.name not in LIBRARY_FILES:
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"Deleted file: library/{item.name}")
                    else:
                        shutil.rmtree(item)
                        print(f"Deleted directory: library/{item.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting library/{item.name}: {e}")
    
    # 处理测试目录中的文件
    tests_path = PROJECT_ROOT / "tests"
    if tests_path.exists():
        for item in tests_path.iterdir():
            rel_path = f"tests/{item.name}"
            if rel_path not in keep_set:
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"Deleted file: {rel_path}")
                    else:
                        shutil.rmtree(item)
                        print(f"Deleted directory: {rel_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {rel_path}: {e}")
    
    # 处理工具目录中的文件
    tools_path = PROJECT_ROOT / "tools"
    if tools_path.exists():
        for item in tools_path.iterdir():
            rel_path = f"tools/{item.name}"
            if rel_path not in keep_set:
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"Deleted file: {rel_path}")
                    else:
                        shutil.rmtree(item)
                        print(f"Deleted directory: {rel_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {rel_path}: {e}")
    
    # 处理网络目录中的文件
    networks_path = PROJECT_ROOT / "networks"
    if networks_path.exists():
        for item in networks_path.iterdir():
            rel_path = f"networks/{item.name}"
            if rel_path not in keep_set:
                try:
                    if item.is_file():
                        item.unlink()
                        print(f"Deleted file: {rel_path}")
                    else:
                        shutil.rmtree(item)
                        print(f"Deleted directory: {rel_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {rel_path}: {e}")
    
    print(f"\nCleanup completed. {deleted_count} items deleted.")
    print("You can now remove the cleanup_unrelated_files.py script if you no longer need it.")

if __name__ == "__main__":
    main()
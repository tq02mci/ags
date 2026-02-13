#!/usr/bin/env python3
"""
A股量化交易系统 - 启动脚本
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent


def check_env():
    """检查环境"""
    print("🔍 检查环境...")

    # 检查 Python 版本
    if sys.version_info < (3, 9):
        print("❌ Python 版本需要 >= 3.9")
        return False

    # 检查 .env 文件
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("⚠️  .env 文件不存在，已复制 .env.example")
        example_file = PROJECT_ROOT / ".env.example"
        if example_file.exists():
            env_file.write_text(example_file.read_text())
        print("⚠️  请编辑 .env 文件配置你的 Supabase 信息")
        return False

    # 检查依赖
    try:
        import pandas
        import fastapi
        import streamlit
        print("✅ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("📦 请运行: pip install -r requirements.txt")
        return False


def init_db():
    """初始化数据库"""
    print("📊 初始化数据库...")
    try:
        # 导入并运行数据同步
        from scripts.sync_data import main as sync_main
        sync_main()
        print("✅ 数据库初始化完成")
        return True
    except Exception as e:
        print(f"❌ 数据库初始化失败: {e}")
        return False


def start_api():
    """启动 API 服务"""
    print("🚀 启动 API 服务...")
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    subprocess.run(cmd)


def start_dashboard():
    """启动 Streamlit 仪表板"""
    print("📈 启动 Streamlit 仪表板...")
    cmd = [
        sys.executable, "-m", "streamlit",
        "run", "src/api/dashboard.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]
    subprocess.run(cmd)


def start_jupyter():
    """启动 Jupyter Notebook"""
    print("📓 启动 Jupyter Notebook...")
    cmd = [
        sys.executable, "-m", "jupyter", "notebook",
        "--ip=0.0.0.0",
        "--port=8888",
        "--no-browser",
        "--allow-root",
        "--NotebookApp.token=''",
        "--NotebookApp.password=''"
    ]
    subprocess.run(cmd)


def run_tests():
    """运行测试"""
    print("🧪 运行测试...")
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]
    subprocess.run(cmd)


def check_data():
    """检查数据质量"""
    print("🔍 检查数据质量...")
    try:
        from scripts.data_quality import main as quality_main
        quality_main()
    except Exception as e:
        print(f"❌ 数据检查失败: {e}")


def show_menu():
    """显示交互式菜单"""
    print("""
╔══════════════════════════════════════════╗
║     📈 A股量化交易系统 - 启动菜单        ║
╠══════════════════════════════════════════╣
║  1. 启动 API 服务 (FastAPI)              ║
║  2. 启动 可视化界面 (Streamlit)          ║
║  3. 启动 Jupyter Notebook                ║
║  4. 初始化数据库                         ║
║  5. 检查数据质量                         ║
║  6. 运行测试                             ║
║  0. 退出                                 ║
╚══════════════════════════════════════════╝
    """)

    choice = input("请选择操作 [0-6]: ").strip()
    return choice


def main():
    parser = argparse.ArgumentParser(description="A股量化交易系统启动脚本")
    parser.add_argument(
        "command",
        choices=["api", "dashboard", "jupyter", "init", "check", "test", "menu"],
        nargs="?",
        default="menu",
        help="要执行的命令"
    )

    args = parser.parse_args()

    # 检查环境
    if args.command != "menu":
        if not check_env():
            return 1

    # 执行命令
    if args.command == "menu":
        while True:
            choice = show_menu()

            if choice == "0":
                print("👋 再见!")
                break
            elif choice == "1":
                if check_env():
                    start_api()
            elif choice == "2":
                if check_env():
                    start_dashboard()
            elif choice == "3":
                if check_env():
                    start_jupyter()
            elif choice == "4":
                if check_env():
                    init_db()
            elif choice == "5":
                if check_env():
                    check_data()
            elif choice == "6":
                if check_env():
                    run_tests()
            else:
                print("❌ 无效选择")

    elif args.command == "api":
        start_api()
    elif args.command == "dashboard":
        start_dashboard()
    elif args.command == "jupyter":
        start_jupyter()
    elif args.command == "init":
        init_db()
    elif args.command == "check":
        check_data()
    elif args.command == "test":
        run_tests()

    return 0


if __name__ == "__main__":
    sys.exit(main())

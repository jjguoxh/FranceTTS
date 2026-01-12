import os
import sys
import platform
import subprocess
import requests
from urllib.parse import urljoin

# ================= 配置 =================
CUDA_VERSION = "cu124"
PYPI_MIRROR = "https://download.pytorch.org/whl/cu124/"
LOCAL_DIR = "./torch_wheels"
os.makedirs(LOCAL_DIR, exist_ok=True)

def get_python_version():
    return f"{sys.version_info.major}{sys.version_info.minor}"

def get_platform_tag():
    system = platform.system().lower()
    if system == "windows":
        return "win_amd64"
    elif system == "linux":
        return "linux_x86_64"
    elif system == "darwin":
        return "macosx_10_9_x86_64"  # Intel Mac
    else:
        raise OSError("Unsupported OS")

def download_with_resume(url, local_path):
    """支持断点续传的下载函数"""
    headers = {}
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        headers["Range"] = f"bytes={file_size}-"
        print(f".Resume from {file_size} bytes...")
    else:
        file_size = 0

    try:
        with requests.get(url, headers=headers, stream=True, timeout=30) as r:
            r.raise_for_status()
            mode = "ab" if file_size > 0 else "wb"
            with open(local_path, mode) as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        # 可选：显示进度
                        # print(f"\rDownloaded: {os.path.getsize(local_path)} bytes", end="")
        print(f"\n✅ Downloaded: {os.path.basename(local_path)}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        sys.exit(1)

def main():
    py_ver = get_python_version()
    plat_tag = get_platform_tag()
    print(f"Detected: Python {py_ver}, Platform {plat_tag}")

    packages = ["torch", "torchvision", "torchaudio"]
    wheel_files = []

    for pkg in packages:
        # 构造文件名（示例：torch-2.4.0+cu124-cp311-cp311-win_amd64.whl）
        # 我们先获取页面列表，找最新版本
        index_url = PYPI_MIRROR
        try:
            resp = requests.get(index_url, timeout=10)
            resp.raise_for_status()
        except:
            print("❌ Failed to fetch package list from Tsinghua mirror.")
            sys.exit(1)

        # 简单匹配（实际可解析 HTML，但这里用字符串匹配足够）
        lines = resp.text.splitlines()
        target_file = None
        for line in lines:
            if f"{pkg}-" in line and CUDA_VERSION in line and f"cp{py_ver}" in line and plat_tag in line:
                # 提取文件名
                start = line.find('href="') + 6
                end = line.find('"', start)
                filename = line[start:end]
                target_file = filename
                break

        if not target_file:
            print(f"❌ No matching wheel found for {pkg} (Python {py_ver}, {plat_tag})")
            sys.exit(1)

        url = urljoin(PYPI_MIRROR, target_file)
        local_file = os.path.join(LOCAL_DIR, target_file)
        wheel_files.append(local_file)

        print(f"📥 Downloading {target_file} ...")
        download_with_resume(url, local_file)

    # 安装
    print("\n📦 Installing packages...")
    for whl in wheel_files:
        cmd = [sys.executable, "-m", "pip", "install", whl]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Failed to install {whl}")
            sys.exit(1)

    print("\n🎉 All packages installed successfully!")
    print("Verify with: python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"")

if __name__ == "__main__":
    main()
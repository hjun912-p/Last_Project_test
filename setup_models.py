import os
import sys
import shutil
import subprocess
import urllib.request
from pathlib import Path

def report_progress(block_num, block_size, total_size):
    if total_size > 0:
        percent = min(100, int(block_num * block_size * 100 / total_size))
        downloaded = block_num * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r   > 다운로드 중... {percent}% ({downloaded:.1f}MB / {total:.1f}MB)")
        sys.stdout.flush()

def clone_repo(url, dest_path, check_file):
    """소스코드 클론. check_file이 이미 있으면 스킵."""
    dest = Path(dest_path)
    if (dest / check_file).exists():
        print(f"   [이미 존재함] {dest.name} 소스코드")
        return True

    print(f"   [클론 시작] {url}")
    try:
        if dest.exists():
            # 빈 폴더만 있는 경우 git clone이 실패하므로 임시 경로에 클론 후 병합
            tmp = dest.parent / (dest.name + "_tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            subprocess.run(["git", "clone", "--depth=1", url, str(tmp)], check=True)
            for item in tmp.iterdir():
                target = dest / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.copytree(str(item), str(target), dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(item), str(target))
                else:
                    shutil.move(str(item), str(target))
            shutil.rmtree(tmp)
        else:
            subprocess.run(["git", "clone", "--depth=1", url, str(dest)], check=True)
        print(f"   [완료] {dest.name} 소스코드")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   [오류] 클론 실패: {e}")
        return False

def download_file(url, save_path):
    save_path = Path(save_path)
    if save_path.exists():
        print(f"   [이미 존재함] {save_path.name}")
        return

    print(f"   [다운로드 시작] {save_path.name}")
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(save_path), report_progress)
        print(f"\n   [완료] {save_path.name}")
    except Exception as e:
        print(f"\n   [오류 발생] {e}")

if __name__ == "__main__":
    print("="*60)
    print(" InSIGHT AI 모델 자동 설치 스크립트")
    print("="*60)

    # 1단계: 소스코드 클론
    REPOS = [
        {
            "name": "VideoFACT 소스코드",
            "url": "https://github.com/ductai199x/videofact-wacv-2024",
            "dest": "external/videofact",
            "check_file": "model",   # model/ 폴더가 있으면 이미 클론된 것
        },
        {
            "name": "FreqNet 소스코드",
            "url": "https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection",
            "dest": "external/freqnet",
            "check_file": "networks",  # networks/ 폴더가 있으면 이미 클론된 것
        },
    ]

    print("\n[1단계] 소스코드 설치")
    for repo in REPOS:
        print(f"\n[*] {repo['name']} 설치 중...")
        clone_repo(repo["url"], repo["dest"], repo["check_file"])

    # 2단계: 가중치 파일 다운로드
    MODELS = [
        {
            "name": "VideoFact (Deepfake Detector)",
            "url": "https://www.dropbox.com/scl/fi/euwth7njdi3nj3wi7o8zu/videofact_df.ckpt?rlkey=hwruc4bui47giukx5urlf1p5j&dl=1",
            "path": "external/videofact/weights/videofact_df.ckpt"
        },
        {
            "name": "FreqNet (Frequency-domain Detector)",
            "url": "https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection/raw/main/4-classes-freqnet-v2.pth",
            "path": "external/freqnet/4-classes-freqnet-v2.pth"
        }
    ]

    print("\n[2단계] 모델 가중치 다운로드")
    for model in MODELS:
        print(f"\n[*] {model['name']} 설치 중...")
        download_file(model['url'], model['path'])

    print("\n" + "="*60)
    print(" 모든 모델이 성공적으로 설치되었습니다.")
    print(" 이제 'python app.py'를 실행하여 앱을 시작하세요.")
    print("="*60)

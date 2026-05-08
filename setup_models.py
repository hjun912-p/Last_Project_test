import os
import sys
import urllib.request
from pathlib import Path

def report_progress(block_num, block_size, total_size):
    """다운로드 진행률 표시"""
    if total_size > 0:
        percent = min(100, int(block_num * block_size * 100 / total_size))
        downloaded = block_num * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r   > 다운로드 중... {percent}% ({downloaded:.1f}MB / {total:.1f}MB)")
        sys.stdout.flush()

def download_file(url, save_path):
    """파일 다운로드 실행"""
    save_path = Path(save_path)
    if save_path.exists():
        print(f"   [이미 존재함] {save_path.name}")
        return

    print(f"   [다운로드 시작] {save_path.name}")
    try:
        # 폴더 생성
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # 다운로드
        urllib.request.urlretrieve(url, str(save_path), report_progress)
        print(f"\n   [완료] {save_path.name}")
    except Exception as e:
        print(f"\n   [오류 발생] {e}")

if __name__ == "__main__":
    print("="*60)
    print(" InSIGHT AI 모델 자동 설치 스크립트")
    print("="*60)

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

    for model in MODELS:
        print(f"\n[*] {model['name']} 설치 중...")
        download_file(model['url'], model['path'])

    print("\n" + "="*60)
    print(" 모든 모델이 성공적으로 설치되었습니다.")
    print(" 이제 'python app.py'를 실행하여 앱을 시작하세요.")
    print("="*60)

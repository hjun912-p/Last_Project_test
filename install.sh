#!/bin/bash
# ================================================================
# Team Environment Setup Script
# Usage: bash install.sh [cuda121 | cuda118 | cpu]
# ================================================================

set -e

CUDA_VER=${1:-cuda121}

echo "=== [1/5] pip 업그레이드 ==="
pip install --upgrade pip

echo "=== [2/5] PyTorch 설치 (${CUDA_VER}) ==="
case $CUDA_VER in
  cuda121)
    pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu121
    ;;
  cuda118)
    pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu118
    ;;
  cpu)
    pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cpu
    ;;
  *)
    echo "Unknown CUDA version: $CUDA_VER"
    echo "Usage: bash install.sh [cuda121 | cuda118 | cpu]"
    exit 1
    ;;
esac

echo "=== [3/5] requirements.txt 설치 ==="
pip install -r requirements.txt

echo "=== [4/5] xformers 설치 (파인튜닝 필요 시) ==="
echo "  xformers는 torch 버전 호환 필요. 수동 설치:"
echo "  pip install xformers==0.0.35"

echo "=== [5/5] 설치 확인 ==="
python -c "
import torch, tensorflow, transformers, langchain, cv2, ultralytics
print(f'torch       : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'tensorflow  : {tensorflow.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'langchain   : {langchain.__version__}')
print(f'opencv      : {cv2.__version__}')
print(f'ultralytics : {ultralytics.__version__}')
"

echo ""
echo "=== 설치 완료 ==="
echo "[추가 주의사항]"
echo "  - KoNLPy: JDK 1.7+ 설치 필요 → https://konlpy.org/ko/latest/install/"
echo "  - Windows: install_windows_extras.bat 별도 실행"

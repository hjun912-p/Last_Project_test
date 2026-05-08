# VideoFact (WACV 2024) 다운로드 기록

## 1. 레포지토리 정보
- **URL**: https://github.com/ductai199x/videofact-wacv-2024
- **설치 경로**: `./external/videofact`
- **다운로드 날짜**: 2026-05-08

## 2. 주요 다운로드 파일 목록
- `external/videofact/model/`: VideoFact 네트워크 아키텍처 정의 파일들
- `external/videofact/utils.py`: 프레임 전처리 및 유틸리티 함수
- `external/videofact/configs/`: 모델 설정 (YAML)
- `external/videofact/inference_single.py`: 개별 파일 추론 로직

## 3. 추가 설치 라이브러리
- `torch`, `torchvision`, `torchaudio` (PyTorch 엔진)
- `PyYAML` (설치 파일 로드용)
- `scipy` (이미지 처리 보조)
- `gdown` (가중치 파일 다운로드용)

## 4. 모델 가중치 상태
- `external/videofact/weights/videofact_df.ckpt`: 다운로드 완료 (890MB)

---

# FreqNet (AAAI 2024) 다운로드 기록

## 1. 레포지토리 정보
- **URL**: https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection
- **설치 경로**: `./external/freqnet`
- **다운로드 날짜**: 2026-05-08
- **기술**: 주파수 영역 학습(Frequency Space Domain Learning)을 통한 일반화 성능 강화 탐지기

## 2. 주요 다운로드 파일 목록
- `external/freqnet/networks/`: 주파수 분석 네트워크 구조 정의
- `external/freqnet/4-classes-freqnet-v2.pth`: 사전 학습된 모델 가중치 (약 7.1MB)
- `external/freqnet/util.py`: 주파수 변환 및 전처리 유틸리티

## 3. 추가 설치 라이브러리
- `albumentations` (이미지 증강 및 전처리)
- `opencv-python-headless` (서버 환경용 OpenCV)
- `pytorch-lightning` (일부 의존성 공유)

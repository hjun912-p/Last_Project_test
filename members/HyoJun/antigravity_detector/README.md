# 🌌 Antigravity AI Video Detector Guide

이 문서는 `antigravity_detector.ipynb` 노트북의 구조와 작동 원리, 그리고 포함된 핵심 코드에 대해 설명합니다. 이 시스템은 인스타그램 미디어를 분석하여 여러 최첨단 탐지 기술을 통해 AI 생성 여부를 판별합니다.

## 🚀 개요
**Antigravity** 시스템은 단순한 분석을 넘어, 인스타그램 게시물/릴스 URL만으로 미디어를 자동 획득하고 다음의 3단계 정밀 검사를 수행합니다.

1. **C2PA (Content Provenance and Authenticity):** 콘텐츠 출처 및 생성 도구 메타데이터 검사
2. **Google SynthID:** Google Gemini 등에서 삽입하는 비가시적 주파수 워터마크 탐지
3. **Meta VideoSeal:** Meta(Facebook) Research의 최신 비디오 워터마크 탐지 기술

---

## 🛠 주요 모듈 및 코드 설명

### 1. 환경 설정 및 유틸리티 로드
기존 `insta-ai-checker` 엔진의 유틸리티를 활용하기 위해 경로를 설정하고 필요한 모듈을 가져옵니다.

```python
# insta-ai-checker 경로 추가 및 모듈 임포트
parent_dir = os.path.abspath("..")
checker_dir = os.path.join(parent_dir, "insta-ai-checker")
sys.path.append(checker_dir)

from utils.downloader import download_instagram_media
from utils.c2pa_checker import check_c2pa
from utils.synthid_checker import check_synthid
```

### 2. VideoSeal 탐지 엔진 (핵심 로직)
Meta의 VideoSeal 모델을 로드하고 분석하는 로직입니다. 라이브러리 내부의 경로 문제를 해결하기 위해 설정 파일을 수동으로 패치하고, 모델 가중치를 자동으로 다운로드하는 기능을 포함합니다.

* **프레임 샘플링:** 비디오 전체에서 **30개의 프레임**을 균등하게 추출하여 분석합니다.
* **판정 기준:** 30개 샘플 중 **6개(20%) 이상**에서 워터마크 신호가 탐지될 경우 최종 'AI 판정'을 내립니다.

```python
def check_videoseal(file_path):
    # ... (생략: 경로 설정 및 설정 로드) ...
    
    # 1. Attenuation 패치: JND 로직 우회 (경로 에러 방지)
    config.args.attenuation = "none"
            
    # 2. 체크포인트 자동 다운로드 및 로드
    if str(ckpt_url).startswith("http"):
        # 원격 서버에서 y_256b_img.pth 다운로드 로직
        # ...
    
    # 3. 30개 프레임 전수 조사
    check_points = np.linspace(0, total_frames - 1, 30, dtype=int)
    for idx in check_points:
        # 프레임 추출 및 모델 추론 (model.detect)
        # 로짓값에 대한 시그모이드 판별 처리 포함
```

### 3. 통합 분석 프로세스 (`run_antigravity_detection`)
모든 탐지 도구를 순차적으로 실행하고 결과를 종합하여 리포트를 생성합니다.

```python
def run_antigravity_detection(url):
    # 1. 미디어 다운로드 (instaloader 활용)
    file_path, media_type = download_instagram_media(url, target_dir)
    
    # 2. 3단계 분석 실행
    c2pa_results = check_c2pa(file_path)
    synth_results = check_synthid(file_path, media_type)
    vseal_results = check_videoseal(file_path)
    
    # 3. 결과 요약 출력 및 SynthID 스펙트럼 시각화 이미지 표시
```

---

## 📊 결과 해석 가이드

* **✅ 탐지됨 (Success):** 해당 기술의 워터마크나 메타데이터가 확실히 발견됨을 의미합니다.
* **❌ 미탐지 (Not Detected):** 워터마크가 없거나, SNS 업로드 과정에서 데이터가 손실되었을 가능성이 있습니다.
* **SynthID 신뢰도:** 65% 이상의 신뢰도가 여러 프레임에서 공통적으로 나타날 때 AI 생성물로 간주합니다.
* **VideoSeal 점수:** 평균 탐지 점수가 높을수록 Meta 계열 AI 도구로 생성되었을 확률이 높습니다.

---

## 📦 요구 사항
노트북 상단의 설치 셀을 통해 다음 라이브러리를 준비해야 합니다:
* `instaloader`, `c2pa-python`, `videoseal`, `torch`, `opencv-python`, `omegaconf` 등

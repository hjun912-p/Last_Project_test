# 🛡️ AI Generated Video Detection Framework (Prototype)

인스타그램에서 수집한 영상의 AI 생성 여부를 판별하기 위한 2단계 필터링 및 MLLM 기반 분석 프로토타입입니다. 
비가시적 워터마크 탐지와 Google Gemini MLLM의 멀티모달 추론 능력을 결합하여 신뢰도 높은 탐지 워크프레임을 제안합니다.

## 📋 프로젝트 개요
최근 생성형 AI 기술의 발전으로 인해 정교해진 AI 생성 영상을 효과적으로 감지하기 위해, 단순한 패턴 매칭을 넘어선 다각도 분석 프로세스를 구축했습니다. 인스타그램 링크로부터 직접 영상을 수집하고, 단계별 필터링을 거쳐 최종 리포트를 생성합니다.

## ⚙️ 주요 기능 및 워크프레임

본 프로젝트는 다음과 같은 **4단계 워크프레임**으로 작동합니다.

### 1. 1차 필터링 (Watermark & Signature Detection)
* **비가시적 워터마크 감지**: 영상 내부에 숨겨진 특정 AI 모델의 시그니처나 디지털 워터마크를 스캔합니다.
* **데이터 전처리**: 수집된 영상에서 분석에 적합한 프레임을 추출하고 노이즈를 제거합니다.

### 2. 2차 필터링 (Gemini MLLM Analysis)
* **멀티모달 분석**: Google Gemini MLLM을 활용하여 영상의 시각적 부자연스러움(텍스처 왜곡, 물리 법칙 불일치 등)을 분석합니다.
* **교차 검증**: 1차 필터링 결과와 MLLM의 시각적 판단 결과를 대조합니다.

### 3. 심층 분석 (Deep Analysis)
* **확률 점수 산출**: 영상의 각 구간별 AI 생성 확률을 계산합니다.
* **XAI (설명 가능한 AI)**: 어떤 부분에서 AI 생성 징후가 포착되었는지 시각화 및 텍스트 데이터로 정리합니다.

### 4. 최종 추론 (Inference & Reporting)
* **결과 도출**: 종합적인 분석 데이터를 바탕으로 해당 영상의 AI 생성 여부를 최종 판별합니다.
* **자동 리포트 생성**: `report.md` 형식의 분석 결과와 주요 프레임 샷을 포함한 결과물을 제공합니다.

## 🛠 Tech Stack
* **Language**: Python 3.x
* **Environment**: Google Colab (GPU T4)
* **Model**: Google Gemini Pro Vision (MLLM)
* **Tools**: Instaloader, OpenCV, PIL
* **Output**: JSON Result, Markdown Report, XAI Visualizations

## 🚀 시작하기

1.  **환경 설정**: Google Colab에서 노트북을 실행하고 필요한 라이브러리를 설치합니다.
2.  **API 키 설정**: Google AI Studio에서 발급받은 Gemini API 키를 입력합니다.
3.  **데이터 수집**: 분석을 원하는 인스타그램 영상 URL을 입력합니다.
4.  **워크플로우 실행**: 데이터 수집부터 최종 추론까지의 셀을 순차적으로 실행합니다.
5.  **결과 확인**: `/content/ai_video_detection_outputs` 폴더에 생성된 리포트와 분석 이미지를 확인합니다.

## 📂 파일 구조
* `instaloader.ipynb`: 전체 탐지 프로세스가 포함된 메인 노트북
* `detection_results.json`: 수치화된 탐지 데이터
* `report.md`: 시각적 분석이 포함된 최종 결과 리포트

---
*본 프로젝트는 AI 생성 영상 탐지 기술의 가능성을 확인하기 위한 프로토타입 단계입니다.*

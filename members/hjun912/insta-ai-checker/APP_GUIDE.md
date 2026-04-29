# 📸 Instagram AI Media Detector Web App Guide

이 문서는 Gradio를 기반으로 구축된 웹 인터페이스 서비스인 `app.py`의 구조와 작동 원리를 설명합니다. 이 애플리케이션은 인스타그램 URL을 통해 미디어를 분석하고 시각적인 리포트를 웹 브라우저에 제공합니다.

## 🚀 개요
`app.py`는 `insta-ai-checker` 시스템의 웹 프론트엔드 역할을 수행합니다. 사용자가 인스타그램 URL을 입력하면 서버 사이드에서 미디어를 처리하고, C2PA, SynthID, Gemini AI의 분석 결과를 실시간으로 업데이트하여 보여줍니다.

---

## 🛠 주요 기능 및 코드 설명

### 1. 사용자 인터페이스 (Gradio UI)
`gr.Blocks`를 사용하여 레이아웃을 구성하며, 탭(Tabs) 형식을 통해 단계별 분석 결과를 깔끔하게 분리하여 보여줍니다.

* **입력부:** 인스타그램 URL 텍스트 박스와 분석 시작 버튼.
* **출력부:** 다운로드된 미디어 파일, 종합 요약, 그리고 각 단계별 상세 분석 탭.

```python
with gr.Blocks(title="AI Media Detector for Instagram") as demo:
    gr.Markdown("# 📸 AI 생성 미디어 판별 서비스 (Instagram)")
    
    with gr.Row():
        url_input = gr.Textbox(label="Instagram URL (Post/Reel)")
        submit_btn = gr.Button("분석 시작", variant="primary")
    
    with gr.Tabs():
        with gr.TabItem("종합 요약"):
            summary_output = gr.Markdown()
        # ... (중략: C2PA, SynthID, Gemini 탭 구성) ...
```

### 2. 분석 핵심 로직 (`process_url`)
사용자의 클릭 이벤트 발생 시 실행되는 메인 함수입니다. `utils` 폴더에 정의된 독립적인 체커들을 호출하여 결과를 취합합니다.

* **Step 0:** `download_instagram_media`를 통해 로컬 `temp` 폴더로 미디어 확보.
* **Step 1:** `check_c2pa`로 메타데이터 기반 생성 도구 확인.
* **Step 2:** `check_synthid`로 Google의 비가시적 워터마크 및 스펙트럼 이상 탐지.
* **Step 3:** `analyze_with_gemini`를 통해 Gemini Vision 모델이 시각적 특징(AI 특유의 왜곡 등) 분석.

```python
def process_url(url):
    # 미디어 다운로드
    file_path, media_type = download_instagram_media(url, "temp")
    
    # 각 도구별 분석 수행
    c2pa_found, c2pa_result = check_c2pa(file_path)
    synth_found, synth_result, synth_viz = check_synthid(file_path, media_type)
    gemini_result = analyze_with_gemini(file_path, media_type)
    
    # 최종 요약 메시지 생성 및 결과 반환
    # ...
```

---

## 📊 인터페이스 구성 요소

1. **종합 요약:** 3단계 분석 결과를 종합하여 최종적인 AI 판정 여부를 Markdown 형식으로 강조하여 보여줍니다.
2. **C2PA 결과:** 이미지/영상 헤더에 포함된 생성 도구 정보를 텍스트로 표시합니다.
3. **SynthID 결과:** Google의 워터마크 탐지 수치와 함께, 주파수 영역 분석 이미지(스펙트럼 시그니처)를 시각적으로 보여줍니다.
4. **Gemini AI 상세 분석:** AI가 사진을 직접 보고 판단한 상세 의견(조명 어색함, 질감 등)을 제공합니다.

---

## 🏃 실행 방법
터미널에서 다음 명령어를 입력하여 로컬 서버를 구동할 수 있습니다:

```bash
python app.py
```
실행 후 터미널에 표시되는 `http://127.0.0.1:7860` 주소로 접속하면 서비스를 이용할 수 있습니다.

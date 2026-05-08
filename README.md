<p align="center">
  <img src="logo/404_HumanNotFound.png" alt="InSIGHT Logo" width="200"/>
</p>

# InSIGHT — AI 생성 콘텐츠 탐지기

> Instagram · YouTube URL 또는 이미지 파일을 입력하면 3단계 분석으로 AI 생성 여부를 판별합니다.

---

## 팀 구성

| 역할 | 이름 |
|------|------|
| 팀장 | 이지수 |
| 팀원 | 박효준 |
| 팀원 | 진민경 |
| 팀원 | 김성일 |
| 팀원 | 신우철 |

---

## 탐지 파이프라인

```
입력 (Instagram URL / YouTube URL / 이미지 파일)
        ↓
   ── Stage 1 (무료, 로컬) ───────────────────
   C2PA 출처 서명 검증
   EXIF AI 도구 흔적 확인
   SynthID 역공학 워터마크 탐지 (CVR + 위상 분석)
        ↓
   ── Stage 2 (Vertex AI) ────────────────────
   Google 공식 SynthID Detector API
        ↓
   ── Stage 3 (Gemini 2.5 Flash) ────────────
   멀티모달 시각 분석 (증거 기반 판정)
        ↓
   ❌ AI 생성 / ✅ 실제 이미지 / ❓ 불확실
```

> 자세한 파이프라인 설명 → [PIPELINE.md](PIPELINE.md)

---

## 빠른 시작 (로컬 실행)

### 1단계 — 저장소 클론

```bash
git clone https://github.com/YOUR_REPO/Last_Project_test.git
cd Last_Project_test
```

### 2단계 — conda 환경 생성 및 활성화

```bash
conda env create -f environment_app.yml
conda activate insight
```

### 3단계 — 모델 가중치 설치 (필수)

이 프로젝트는 고성능 AI 모델을 사용합니다. 용량이 커서 깃허브에 직접 올리지 못하니, 설치 후 아래 명령어를 입력해 모델을 한 번에 다운로드하세요:

```bash
python setup_models.py
```

### 4단계 — 앱 실행

```bash
python app.py
```

브라우저에서 `http://localhost:7860` 접속

---

## 프로젝트 파일 구조

```
Last_Project_test/
├── app.py                   # 메인 앱 실행 파일
├── synthid_detector.py      # SynthID 역공학 탐지 모듈
├── synthid_vertex.py        # Vertex AI SynthID 연동 모듈
├── requirements_app.txt     # 앱 전용 pip 패키지 목록
├── environment_app.yml      # conda 환경 설정 파일
├── .env.example             # 환경변수 예시 (복사 후 .env로 사용)
├── .env                     # 실제 환경변수 (git 제외, 직접 생성)
├── README.md                # 프로젝트 소개 (현재 파일)
├── PIPELINE.md              # 탐지 파이프라인 설명
├── members/                 # 팀원별 작업 폴더
│   ├── woochul/
│   ├── hjun912/
│   ├── jinmg/
│   ├── JISOO/
│   └── tjddlf/
└── test_dataset/            # 테스트 이미지 (Google Drive에서 다운로드)
```

---

## 주요 파일 설명

| 파일 | 설명 |
|------|------|
| `app.py` | Gradio 기반 메인 앱. 로컬에서 `python app.py` 로 실행 |
| `synthid_detector.py` | 역공학 SynthID 워터마크 탐지 (로컬, 무료) |
| `synthid_vertex.py` | Vertex AI 공식 SynthID Detector 연동 |
| `environment_app.yml` | 팀원 전체 공통 conda 환경 (Python 3.11) |
| `requirements_app.txt` | pip만 사용할 경우의 패키지 목록 |
| `.env.example` | 환경변수 템플릿. 복사해서 `.env` 로 사용 |

---

## 분석 단계별 필요 설정

| 단계 | 필요 설정 | 비용 |
|------|----------|------|
| Stage 1 — 메타데이터/SynthID 역공학 | 없음 | 무료 |
| Stage 2 — Vertex AI SynthID | `GOOGLE_APPLICATION_CREDENTIALS` | 약 $0.0002/장 |
| Stage 3 — Gemini 2.5 Flash | `GEMINI_API_KEY` | 무료 티어 있음 |

> Stage 2, 3 설정이 없어도 앱은 정상 실행됩니다. 해당 단계만 스킵됩니다.

---

## 테스트 데이터셋

이미지 데이터셋은 용량 문제로 Google Drive에서 관리합니다.

**Google Drive:** https://drive.google.com/drive/folders/11Hu7vRj2f-l6ottcFQLy68xRHLheUs4w

| 폴더 | 내용 | 출처 |
|------|------|------|
| `test_dataset/ai/` | AI 생성 이미지 | Civitai API (Flux, SDXL 등) |
| `test_dataset/real/` | 실제 이미지 | randomuser.me, Unsplash |

다운로드 후 프로젝트 루트의 `test_dataset/` 폴더에 배치하세요.

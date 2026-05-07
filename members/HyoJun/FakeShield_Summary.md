# 🛡️ FakeShield: 설명 가능한 이미지 위조 탐지 종합 가이드

본 문서는 **ICLR 2025**에 발표된 논문 **"FakeShield: A Multimodal Framework for Explainable Image Forgery Detection and Localization"**와 [공식 GitHub 저장소](https://github.com/zhipeixu/FakeShield)의 내용을 바탕으로 작성되었습니다.

---

## 🌟 1. 핵심 요약: "왜 이 모델이 특별한가?"
기존 모델들이 "위조입니다"라는 결과만 던져주었다면, **FakeShield**는 **"어디가, 어떻게, 왜 가짜인지"**를 사람처럼 설명해주는 차세대 위조 탐지 모델입니다.

- **설명 가능한 AI (e-IFDL):** 위조 여부 판별 + 정밀 위치 식별 + 논리적 근거 제시.
- **도메인 갈등 해결:** 포토샵, 딥페이크, AI 생성물 등 서로 다른 위조 특성을 '도메인 태그'로 구분하여 학습.
- **압도적 성능:** IMD2020 등 주요 벤치마크에서 기존 SOTA 대비 정확도 0.08, F1 0.05 향상.

---

## 🧠 2. 어떻게 작동하나요? (아키텍처)

FakeShield는 생각하는 부분(**DTE-FDM**)과 눈으로 정밀하게 따내는 부분(**MFLM**)이 나누어진 '디커플링' 구조입니다.

### ① 데이터의 힘: MMTD-Set
- **구성:** 이미지 + 위조 마스크 + **GPT-4o가 생성한 정밀 설명**.
- **학습 내용:** "경계선이 뭉개짐", "그림자 방향이 어색함" 같은 논리적 추론법을 학습합니다.

### ② 생각하는 뇌: DTE-FDM (Detection & Explanation)
- **도메인 태그 생성기 (DTG):** "이건 포토샵이네!", "이건 딥페이크네!"라고 먼저 분류하여 모델이 맞춤형으로 분석하게 돕습니다.
- **M-LLM (Llama/LLaVA 기반):** 이미지를 보고 위조 근거를 텍스트로 작성하며, 위치를 찾으라는 신호인 `<SEG>` 토큰을 보냅니다.

### ③ 정밀한 눈: MFLM (Localization)
- **SAM (Segment Anything Model) 활용:** `<SEG>` 토큰의 힌트를 받아, SAM이 픽셀 단위로 아주 정밀하게 조작된 영역의 테두리를 따냅니다.

---

## 🔍 3. 모델이 찾아내는 위조의 증거
| 구분 | 주요 단서 (Artifacts) |
| :--- | :--- |
| **픽셀 수준 (미세 결함)** | 부자연스러운 경계, 해상도 불일치, 노이즈 패턴 차이, 조명/그림자 오류 |
| **이미지 수준 (논리 오류)** | 원근법 위반, 물리 법칙 위배(공중 부양 등), 문맥에 맞지 않는 물체 |

---

## 💻 4. 개발자를 위한 기술 정보 (GitHub 분석)

[GitHub 저장소](https://github.com/zhipeixu/FakeShield)를 통해 직접 모델을 돌려볼 수 있습니다.

### 🛠️ 설치 및 환경 (Setup)
- **OS/언어:** Python 3.9, PyTorch 1.13.0, CUDA 11.6.
- **핵심 라이브러리:** `mmcv v1.4.7` (특수 연산 필요).
- **편의 기능:** 개발 환경 구축이 까다롭기 때문에 저자들이 **Docker 이미지**(`zhipeixu/mflm:v1.0` 등)를 제공합니다.

### 📦 사전 학습 모델 (Checkpoints)
[Hugging Face](https://huggingface.co/zhipeixu/fakeshield-v1-22b)에서 가중치를 다운로드할 수 있습니다.
- `DTE-FDM`, `MFLM` 가중치 및 `DTG.pth` 포함.
- 정밀 마스크 생성을 위한 `SAM (ViT-H)` 가중치 필요.

### 🚀 주요 실행 명령어
- **데모 실행:** `bash scripts/cli_demo.sh` (제공된 이미지로 즉시 테스트 가능)
- **학습:** `scripts/DTE-FDM/finetune_lora.sh` (LoRA 기법으로 효율적인 미세조정)

---

## 🎯 5. 결론 및 활용 가치
FakeShield는 단순히 위조를 잡는 도구를 넘어, **법적 증거 수집(Digital Forensics)**이나 **AI 가이드라인 준수 확인** 등 신뢰가 중요한 분야에서 강력한 힘을 발휘합니다. 

---
*이 문서는 arXiv:2410.02761 논문과 공식 GitHub 소스코드를 분석하여 작성되었습니다.*

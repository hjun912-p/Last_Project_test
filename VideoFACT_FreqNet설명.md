# VideoFact & FreqNet: 딥러닝 기반 고급 탐지 기술 설명

InSIGHT는 최신 학계(WACV, AAAI 2024)에서 발표된 고성능 AI 탐지 모델을 통합하여, 단순 메타데이터 분석을 넘어선 정밀한 검증을 수행합니다.

---

## 1. VideoFact (WACV 2024)
**파일명:** `external/videofact`

### 기술 개요
- **Full Name:** VideoFACT: Detecting Video Forgeries Using Attention, Scene Context, and Forensic Traces
- **발표:** WACV 2024 (IEEE/CVF Winter Conference on Applications of Computer Vision)

### 핵심 분석 원리
1. **디지털 포렌식 흔적 (Forensic Traces)**
   - 이미지 센서에서 발생하는 고유의 노이즈 패턴(PRNU)과 AI 생성물 특유의 픽셀 레벨 아티팩트를 분석합니다.
   - 압축 과정에서 발생하는 미세한 이상 현상을 감지합니다.
2. **장면 문맥 (Scene Context)**
   - 이미지 내의 조명 일관성, 그림자의 물리적 타당성, 객체 간의 기하학적 관계를 분석하여 AI 생성물의 부자연스러움을 잡아냅니다.
3. **어텐션 메커니즘 (Attention)**
   - 이미지 내에서 조작 가능성이 가장 높은 구역(예: 얼굴 경계면, 물체의 가장자리)에 집중하여 정밀 분석합니다.

---

## 2. FreqNet (AAAI 2024)
**파일명:** `external/freqnet`

### 기술 개요
- **Full Name:** Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning
- **발표:** AAAI 2024 (Association for the Advancement of Artificial Intelligence)

### 핵심 분석 원리
1. **주파수 영역 도메인 학습 (Frequency Space Domain Learning)**
   - 이미지를 픽셀 단위(공간 영역)가 아닌 주파수 성분으로 변환(Discrete Cosine Transform 등)하여 분석합니다.
   - 사람이 육안으로 식별할 수 없는 고주파 영역의 미세한 위조 패턴을 탐지합니다.
2. **일반화 성능 (Generalizability)**
   - 특정 생성 AI 모델에만 최적화되지 않고, 다양한 알고리즘(GAN, Diffusion 모델 등)이 공통적으로 남기는 주파수 공간의 왜곡을 찾아냅니다.
3. **위조 흔적 정밀 감지**
   - 생성 AI가 이미지를 업샘플링(Upsampling)하거나 필터링할 때 발생하는 특유의 주파수 불일치를 정확하게 잡아냅니다.

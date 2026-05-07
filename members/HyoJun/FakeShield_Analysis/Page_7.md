# FakeShield 논문 요약 및 번역 - 7페이지

## 4. 실험 (Experiment)

### 4.1 실험 설정 (Experimental Setup)
- **데이터셋**: MMTD-Set을 구축하여 학습 및 테스트에 사용했습니다. 
  - 학습: CASIAv2, Fantastic Reality, FFHQ, FaceApp 등 활용.
  - 테스트: CASIA1+, IMD2020, Columbia, Coverage, DSO, Korus, Seq-DeepFake 등 다양한 공개 벤치마크 활용.
- **비교 모델**: 
  - IFDL 성능: SPAN, MantraNet, OSN, HiFi-Net, PSCC-Net, CAT-Net, MVSS-Net 등.
  - DeepFake 탐지: CADDM, HiFi-DeepFake, RECCE, Exposing 등.
  - 설명 능력: LLaVA, InternVL2, Qwen2-VL, GPT-4o 등.
- **평가 지표**: 탐지(ACC, F1), 위치 식별(IoU, F1), 해석력(CSS - 코사인 의미론적 유사도).

### 4.2 이미지 위조 탐지 방법과의 비교
- **성능 (표 1 참고)**: FakeShield는 Photoshop, DeepFake, AIGC 편집 데이터셋 모두에서 거의 모든 지표에서 가장 높은 정확도를 달성했습니다. 특히 복잡한 Photoshop 변조(CASIA1+, Columbia 등)에서 기존 모델들을 큰 폭으로 앞섰습니다.

---
**표 1 분석**: 다양한 변조 데이터셋에 대한 성능 비교 결과, FakeShield가 모든 항목에서 1위(굵게 표시) 또는 2위(밑줄)를 차지하며 압도적인 일반화 능력을 입증했습니다.
**구현 상세**: NVIDIA A100 GPU 4대를 사용하였으며, M-LLM 학습에는 LoRA를 적용했습니다.

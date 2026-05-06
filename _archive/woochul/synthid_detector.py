"""
SynthID 워터마크 탐지 모듈
출처: github.com/aloshdenny/reverse-SynthID (robust_extractor.py 직접 구현)

핵심 원리:
  실제 Gemini 생성 이미지 291장 분석으로 추출한 캐리어 주파수 사용.
  웨이블렛(db4/sym8/coif3) + 양방향 필터 + NLM 다중 디노이저 융합 후
  노이즈 도메인에서 캐리어 주파수 에너지를 랜덤 위치와 비교(CVR).
  탐지 신뢰도: 워터마크 있음 0.92-0.99 / 없음 0.47-0.53
"""

import numpy as np
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from scipy.fft import fft2, fftshift
except ImportError:
    from numpy.fft import fft2, fftshift


# ── 실제 SynthID 캐리어 주파수 (512px 기준, 291개 Gemini 이미지에서 추출) ──
# 위상 일관성 0.95+ 검증됨
CARRIERS_DARK = [          # 어두운 이미지 / 대각선 격자 패턴
    (-5, -3), (5, 3),   (-5, 3),  (5, -3),
    (-3, -4), (3, 4),   (-3, 4),  (3, -4),
    (-4, -3), (4, 3),   (-4, 3),  (4, -3),
    (-5, -1), (5, 1),   (-5, 1),  (5, -1),
    (-5, -2), (5, 2),   (-5, 2),  (5, -2),
    (-2, -5), (2, 5),   (-2, 5),  (2, -5),
    (-1, -5), (1, 5),   (-1, 5),  (1, -5),
    (-4, -4), (4, 4),   (-4, 4),  (4, -4),
    (-1, -6), (1, 6),   (-3, -5), (3, 5),
]

CARRIERS_WHITE = [         # 밝은 이미지 / 수평축 패턴
    (0, -7),  (0, 7),  (0, -8),  (0, 8),
    (0, -9),  (0, 9),  (0, -10), (0, 10),
    (0, -11), (0, 11), (0, -12), (0, 12),
    (0, -20), (0, 20), (0, -21), (0, 21),
    (0, -22), (0, 22), (0, -23), (0, 23),
]

ALL_CARRIERS = CARRIERS_DARK + CARRIERS_WHITE


# ────────────────────────────────────────────────────────
# 디노이저
# ────────────────────────────────────────────────────────

def _wavelet_denoise(channel: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
    coeffs = pywt.wavedec2(channel, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(channel.size + 1))
    new_coeffs = [coeffs[0]] + [
        tuple(pywt.threshold(d, threshold, mode='soft') for d in detail)
        for detail in coeffs[1:]
    ]
    denoised = pywt.waverec2(new_coeffs, wavelet)
    return denoised[:channel.shape[0], :channel.shape[1]]


def _extract_noise_fused(img_u8: np.ndarray) -> np.ndarray:
    """
    다중 디노이저 융합 노이즈 추출
    웨이블렛(db4/sym8/coif3) 1.0 + 양방향 필터 0.8 + NLM 0.7 가중 평균
    """
    img_f = img_u8.astype(np.float32) / 255.0
    noises, weights = [], []

    if PYWT_AVAILABLE:
        for wv in ['db4', 'sym8', 'coif3']:
            noise = np.zeros_like(img_f)
            for c in range(3):
                denoised = _wavelet_denoise(img_f[:, :, c], wv)
                noise[:, :, c] = img_f[:, :, c] - denoised
            noises.append(noise)
            weights.append(1.0)

    if CV2_AVAILABLE:
        bilateral = np.zeros_like(img_f)
        for c in range(3):
            bilateral[:, :, c] = cv2.bilateralFilter(img_f[:, :, c], 9, 75, 75)
        noises.append(img_f - bilateral)
        weights.append(0.8)

        nlm = cv2.fastNlMeansDenoisingColored(
            img_u8, None, 10, 10, 7, 21
        ).astype(np.float32) / 255.0
        noises.append(img_f - nlm)
        weights.append(0.7)

    if not noises:
        # 폴백: 가우시안 차이
        from PIL import ImageFilter
        pil = Image.fromarray(img_u8)
        blurred = np.array(pil.filter(ImageFilter.GaussianBlur(radius=1))).astype(np.float32) / 255.0
        return img_f - blurred

    total = sum(weights)
    return sum(n * w for n, w in zip(noises, weights)) / total


# ────────────────────────────────────────────────────────
# CVR (Carrier-vs-Random) 비율 계산
# ────────────────────────────────────────────────────────

def _cvr_score(noise: np.ndarray, size: int) -> float:
    """
    노이즈 FFT에서 SynthID 캐리어 주파수 에너지 / 랜덤 위치 에너지 비율
    원본 코드 기준: CVR > 2.0 → 워터마크 의심 (sigmoid 중심 2.0)
    """
    center = size // 2
    noise_gray = np.mean(noise, axis=2).astype(np.float32)
    f_noise = fftshift(fft2(noise_gray))
    noise_mag = np.abs(f_noise)

    carrier_mags = []
    for fy, fx in ALL_CARRIERS:
        y, x = fy + center, fx + center
        if 0 <= y < size and 0 <= x < size:
            carrier_mags.append(noise_mag[y, x])

    rng = np.random.RandomState(42)
    random_mags = []
    for _ in range(len(ALL_CARRIERS) * 4):
        ry = rng.randint(10, size - 10)
        rx = rng.randint(10, size - 10)
        if abs(ry - center) < 5 and abs(rx - center) < 5:
            continue
        random_mags.append(noise_mag[ry, rx])

    if not carrier_mags or not random_mags:
        return 1.0

    return float(np.mean(carrier_mags)) / (float(np.mean(random_mags)) + 1e-10)


# ────────────────────────────────────────────────────────
# 위상 대칭 일관성 (보조 지표)
# ────────────────────────────────────────────────────────

def _phase_symmetry_score(img_gray: np.ndarray, size: int) -> float:
    """
    켤레 대칭 캐리어 쌍의 위상 대칭성 측정
    실제 워터마크는 대칭 쌍이 위상을 공유 → 점수 높음
    """
    center = size // 2
    f = fftshift(fft2(img_gray.astype(np.float32)))
    phase = np.angle(f)

    sym_scores = []
    for fy, fx in CARRIERS_DARK:
        y1, x1 = fy + center, fx + center
        y2, x2 = -fy + center, -fx + center
        if not (0 <= y1 < size and 0 <= x1 < size and
                0 <= y2 < size and 0 <= x2 < size):
            continue
        diff = np.abs(np.angle(np.exp(1j * (phase[y1, x1] + phase[y2, x2]))))
        sym_scores.append(1.0 - diff / np.pi)

    return float(np.mean(sym_scores)) if sym_scores else 0.5


# ────────────────────────────────────────────────────────
# 메인 탐지 함수
# ────────────────────────────────────────────────────────

def detect_synthid(image: Image.Image) -> tuple:
    """
    SynthID 워터마크 탐지

    Returns:
        (True | None, 결과 문자열)
    """
    TARGET_SIZE = 512

    try:
        img_rgb = image.convert("RGB")
        img_resized = img_rgb.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        img_u8 = np.array(img_resized)

        # ── 1. 다중 디노이저 융합 노이즈 추출 ──
        noise = _extract_noise_fused(img_u8)

        # ── 2. CVR 비율 (캐리어 에너지 vs 랜덤) ──
        cvr = _cvr_score(noise, TARGET_SIZE)

        # ── 3. 위상 대칭 일관성 ──
        gray = np.mean(img_u8, axis=2)
        phase_sym = _phase_symmetry_score(gray, TARGET_SIZE)

        # ── 4. 신뢰도 계산 ──
        # CVR sigmoid (원본 코드와 동일, 중심 2.0)
        cvr_conf = float(1.0 / (1.0 + np.exp(-2.0 * (cvr - 2.0))))
        # 위상 대칭 sigmoid (중심 0.62)
        phase_conf = float(1.0 / (1.0 + np.exp(-15.0 * (phase_sym - 0.62))))

        # 가중 합산 (CVR 주도, 위상 보조)
        confidence = float(min(1.0, cvr_conf * 0.75 + phase_conf * 0.25))
        confidence_pct = round(confidence * 100, 1)

        detected = confidence >= 0.60

        # ── 결과 문자열 ──
        detail = (
            f"캐리어/랜덤 비율(CVR)={cvr:.3f}  "
            f"위상대칭={phase_sym:.3f}  "
            f"CVR신뢰={cvr_conf*100:.1f}%  "
            f"위상신뢰={phase_conf*100:.1f}%"
        )

        if detected:
            return True, (
                f"❌ SynthID 워터마크 감지\n"
                f"   탐지 신뢰도: {confidence_pct}%\n"
                f"   {detail}"
            )
        elif confidence >= 0.45:
            return None, (
                f"⚠️ SynthID 경계값 (불확실)\n"
                f"   탐지 신뢰도: {confidence_pct}%\n"
                f"   {detail}"
            )
        else:
            return None, (
                f"🔍 SynthID 미감지\n"
                f"   탐지 신뢰도: {confidence_pct}%\n"
                f"   {detail}"
            )

    except Exception as e:
        return None, f"🔍 SynthID: 분석 오류 — {str(e)}"

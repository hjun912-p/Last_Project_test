import cv2
import numpy as np
from scipy.fftpack import fft2, fftshift
from scipy.stats import kurtosis


class StatisticalRealDefender:
    """Stage 2 guard that checks statistical evidence for real camera images."""

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.kurt_sigma = 22.0
        self.energy_threshold = 210.0
        self.energy_steer = 0.02

    def _gaussian_prob(self, value: float, target: float, sigma: float) -> float:
        return float(np.exp(-((value - target) ** 2) / (2 * (sigma**2))))

    def _sigmoid_prob(self, value: float, center: float, steer: float) -> float:
        return float(1 / (1 + np.exp(steer * (value - center))))

    def analyze(self, pil_image) -> tuple[float, str, dict]:
        img_gray = np.array(pil_image.convert("L"))

        denoised = cv2.medianBlur(img_gray, 3)
        residual = img_gray.astype(np.float32) - denoised.astype(np.float32)
        res_kurt = float(abs(kurtosis(residual.flatten())))
        if not np.isfinite(res_kurt):
            res_kurt = 0.0
        prnu_p = self._gaussian_prob(res_kurt, 0.0, self.kurt_sigma)

        f_shift = fftshift(fft2(img_gray))
        mag = 20 * np.log(np.abs(f_shift) + 1)
        h, w = mag.shape
        hf_energy = float(np.mean(mag[: min(20, h), : min(20, w)]))
        nis_p = self._sigmoid_prob(hf_energy, self.energy_threshold, self.energy_steer)

        if prnu_p > 0.6:
            mode = "High-Quality Forensic Mode"
            total_p = (prnu_p * 0.4) + (nis_p * 0.6)
        else:
            mode = "Compression-Resistant Mode (NIS Focus)"
            total_p = (nis_p * 0.9) + (prnu_p * 0.1)

        details = {
            "prnu_score": float(prnu_p),
            "nis_score": float(nis_p),
            "kurtosis": res_kurt,
            "hf_energy": hf_energy,
        }
        return float(np.clip(total_p, 0.0, 1.0)), mode, details

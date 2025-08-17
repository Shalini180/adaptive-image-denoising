"""
Complete Adaptive Image Denoising System
Integration of all optimized components for production deployment
"""

import numpy as np
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
from scipy import stats
import warnings

from core_methods import CoreDenoisingMethods
from improved_noise_detector import ImprovedNoiseDetector

warnings.filterwarnings("ignore")


def safe_index(array, *indices):
    """Safely index array by converting indices to integers"""
    safe_indices = tuple(
        int(idx) if isinstance(idx, (float, np.floating)) else idx for idx in indices
    )
    return array[safe_indices]


class NoiseDetector:
    """
    Advanced noise type detection using statistical analysis
    """

    def __init__(self):
        self.noise_features_config = {
            "gaussian": {
                "features": ["spatial_uniformity", "normality_test", "variance_consistency"],
                "thresholds": [0.7, 0.05, 0.8],
            },
            "salt_pepper": {
                "features": ["impulse_ratio", "binary_distribution", "spatial_clustering"],
                "thresholds": [0.01, 0.8, 0.3],
            },
            "speckle": {
                "features": ["multiplicative_test", "gamma_fit", "texture_correlation"],
                "thresholds": [0.6, 0.05, 0.7],
            },
            "uniform": {
                "features": ["range_consistency", "uniformity_test", "frequency_distribution"],
                "thresholds": [0.8, 0.05, 0.7],
            },
            "poisson": {
                "features": ["variance_mean_ratio", "poisson_fit", "intensity_dependence"],
                "thresholds": [0.8, 0.05, 0.6],
            },
        }

    def extract_noise_features(self, image):
        """Extract comprehensive noise features for classification"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        features = {}

        # 1. Spatial uniformity
        patches = self._extract_patches(gray, size=16)
        patch_variances = [np.var(cv2.Laplacian(p, cv2.CV_64F)) for p in patches]
        if len(patch_variances) > 1:
            features["spatial_uniformity"] = 1.0 - (
                np.std(patch_variances) / (np.mean(patch_variances) + 1e-8)
            )
        else:
            features["spatial_uniformity"] = 0.5

        # 2. Normality test
        residuals = gray - cv2.GaussianBlur(gray, (5, 5), 1.0)
        _, p_value = stats.normaltest(residuals.flatten())
        features["normality_test"] = 1.0 - min(p_value, 1.0)

        # 3. Impulse noise
        diff = np.abs(cv2.medianBlur(gray, 5).astype(np.float32) - gray.astype(np.float32))
        features["impulse_ratio"] = np.sum(diff > 50) / gray.size

        # 4. Variance consistency
        bright_mask = gray > np.mean(gray)
        dark_mask = gray <= np.mean(gray)
        if np.sum(bright_mask) > 0 and np.sum(dark_mask) > 0:
            bright_var = np.var(residuals[bright_mask])
            dark_var = np.var(residuals[dark_mask])
            features["variance_consistency"] = 1.0 - abs(bright_var - dark_var) / (
                bright_var + dark_var + 1e-8
            )
        else:
            features["variance_consistency"] = 0.5

        # 5. Multiplicative test
        smooth = cv2.GaussianBlur(gray, (9, 9), 2.0)
        ratio = gray.astype(np.float32) / (smooth.astype(np.float32) + 1e-8)
        features["multiplicative_test"] = np.std(ratio) / (np.mean(ratio) + 1e-8)

        # 6. Uniformity test
        hist, _ = np.histogram(residuals.flatten(), bins=50)
        hist_norm = hist / np.sum(hist)
        uniform_hist = np.ones_like(hist_norm) / len(hist_norm)
        features["uniformity_test"] = 1.0 - np.sum(np.abs(hist_norm - uniform_hist)) / 2.0

        # 7. Poisson characteristics
        features["variance_mean_ratio"] = np.var(gray) / (np.mean(gray) + 1e-8)

        return features

    def _extract_patches(self, image, size=16):
        """Extract non-overlapping patches from image"""
        patches = []
        h, w = image.shape
        for i in range(0, h - size, size):
            for j in range(0, w - size, size):
                patch = image[i : i + size, j : j + size]
                if patch.shape == (size, size):
                    patches.append(patch)
        return patches[:20]

    def detect_noise_type(self, image):
        """Detect primary noise type using rule-based classification"""
        features = self.extract_noise_features(image)
        noise_scores = {}

        for noise_type, config in self.noise_features_config.items():
            score, feature_count = 0.0, 0
            for f_name, threshold in zip(config["features"], config["thresholds"]):
                if f_name not in features:
                    continue
                val = features[f_name]
                if noise_type == "gaussian":
                    if f_name == "spatial_uniformity":
                        score += 1.0 if val > threshold else val / threshold
                    elif f_name == "normality_test":
                        score += 1.0 if val < threshold else (1.0 - val)
                    elif f_name == "variance_consistency":
                        score += 1.0 if val > threshold else val / threshold
                elif noise_type == "salt_pepper" and f_name == "impulse_ratio":
                    score += min(val / threshold, 1.0) if val > threshold else 0.0
                elif noise_type == "speckle" and f_name == "multiplicative_test":
                    score += min(val / threshold, 1.0) if val > threshold else 0.0
                elif noise_type == "uniform" and f_name == "uniformity_test":
                    score += 1.0 if val < threshold else (1.0 - val)
                elif noise_type == "poisson" and f_name == "variance_mean_ratio":
                    score += 1.0 - abs(val - 1.0) / 2.0
                feature_count += 1
            noise_scores[noise_type] = score / max(feature_count, 1)

        primary_noise = max(noise_scores, key=noise_scores.get)
        return {
            "primary_noise_type": primary_noise,
            "confidence": noise_scores[primary_noise],
            "all_scores": noise_scores,
            "features": features,
        }

    def estimate_noise_level(self, image):
        """Estimate noise level using robust MAD estimator"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
        return min(sigma / 100.0, 1.0)


class AdaptiveImageDenoiser:
    """
    Complete adaptive denoising pipeline integrating:
    - Noise detection
    - Weighted method combination
    - Uncertainty-based refinement
    - Performance monitoring
    """

    def __init__(self):
        self.detector = ImprovedNoiseDetector()
        self.methods = CoreDenoisingMethods()
        self.log_dir = Path("denoising_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.history_file = self.log_dir / "performance_history.json"
        if not self.history_file.exists():
            with open(self.history_file, "w") as f:
                json.dump([], f)

        # Empirical weights
        self.method_weights = {
            "gaussian": {"nlm": 0.35, "wavelet": 0.25, "bm3d": 0.40},
            "salt_pepper": {"median": 0.45, "adaptive_median": 0.35, "nlm": 0.20},
            "speckle": {"wavelet": 0.40, "bm3d": 0.35, "nlm": 0.25},
            "uniform": {"wavelet": 0.50, "nlm": 0.30, "bm3d": 0.20},
            "poisson": {"bm3d": 0.40, "nlm": 0.35, "wavelet": 0.25},
        }

    def compute_uncertainty_map(self, results):
        """Compute pixel-level uncertainty across methods"""
        stack = np.stack(list(results.values()), axis=-1)
        return np.std(stack.astype(np.float32), axis=-1)

    def adaptive_refinement(self, results, uncertainty_map, noise_info):
        """Refine results adaptively using uncertainty"""
        combined = self.weighted_combination(results, noise_info["primary_noise_type"])
        mask = uncertainty_map > 0.1
        if np.any(mask):
            combined[mask] = results["bm3d"][mask] if "bm3d" in results else combined[mask]
        return combined

    def weighted_combination(self, results, noise_type):
        """Combine results with noise-specific weights"""
        weights = self.method_weights.get(noise_type, {})
        h, w = next(iter(results.values())).shape[:2]
        combined = np.zeros((h, w, 3), dtype=np.float32)
        total = 0.0
        for m, res in results.items():
            if m in weights:
                combined += res.astype(np.float32) * weights[m]
                total += weights[m]
        return np.clip(combined / max(total, 1e-8), 0, 255).astype(np.uint8)

    def denoise_image(self, noisy_image):
        """Full denoising pipeline"""
        start = time.time()

        # Step 1: Noise detection
        noise_info = self.detector.detect_noise(noisy_image)
        level = self.detector.estimate_noise_level(noisy_image)

        # Step 2: Apply methods
        results = {}
        if noise_info["primary_noise_type"] == "gaussian":
            results["nlm"] = self.methods.non_local_means(noisy_image, level)
            results["wavelet"] = self.methods.wavelet_denoising(noisy_image)
            results["bm3d"] = self.methods.bm3d_denoising(noisy_image)
        elif noise_info["primary_noise_type"] == "salt_pepper":
            results["median"] = self.methods.median_filter(noisy_image)
            results["adaptive_median"] = self.methods.adaptive_median_filter(noisy_image)
            results["nlm"] = self.methods.non_local_means(noisy_image, level)
        else:
            results["wavelet"] = self.methods.wavelet_denoising(noisy_image)
            results["bm3d"] = self.methods.bm3d_denoising(noisy_image)
            results["nlm"] = self.methods.non_local_means(noisy_image, level)

        # Step 3: Uncertainty & refinement
        uncertainty_map = self.compute_uncertainty_map(results)
        final = self.adaptive_refinement(results, uncertainty_map, noise_info)

        runtime = time.time() - start
        self._log_results(noisy_image, final, noise_info, level, runtime)

        return final, {
            "noise_info": noise_info,
            "noise_level": level,
            "uncertainty": np.mean(uncertainty_map),
            "runtime": runtime,
        }

    def _log_results(self, noisy_image, denoised_image, noise_info, noise_level, runtime):
        """Log results for monitoring"""
        log = {
            "timestamp": datetime.now().isoformat(),
            "noise_type": noise_info["primary_noise_type"],
            "confidence": float(noise_info["confidence"]),
            "noise_level": float(noise_level),
            "runtime": float(runtime),
            "image_shape": list(noisy_image.shape),
        }
        with open(self.history_file, "r+") as f:
            history = json.load(f)
            history.append(log)
            f.seek(0)
            json.dump(history, f, indent=2)


if __name__ == "__main__":
    img = cv2.imread("test_images/noisy_example.jpg")
    denoiser = AdaptiveImageDenoiser()
    denoised, info = denoiser.denoise_image(img)
    print("Noise type:", info["noise_info"]["primary_noise_type"])
    print("Confidence:", info["noise_info"]["confidence"])
    print("Runtime:", info["runtime"])
    cv2.imwrite("results/denoised_output.jpg", denoised)

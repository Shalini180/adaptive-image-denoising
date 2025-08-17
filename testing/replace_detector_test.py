# Final Balanced Improved Noise Detector
import numpy as np
import cv2
from scipy import ndimage, stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ImprovedNoiseDetector:
    def __init__(self):
        self.noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
        
    def detect_noise(self, image):
        """Balanced noise detection with no bias"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate all detection scores with BALANCED caps
        raw_scores = {}
        
        # BALANCED CAPS - no single type dominates
        raw_scores['gaussian'] = min(self._detect_gaussian_balanced(gray), 0.70)
        raw_scores['salt_pepper'] = min(self._detect_salt_pepper_balanced(gray), 0.75)
        raw_scores['speckle'] = min(self._detect_speckle_balanced(gray), 0.65)
        raw_scores['uniform'] = min(self._detect_uniform_balanced(gray), 0.55)
        raw_scores['poisson'] = min(self._detect_poisson_balanced(gray), 0.70)
        
        # Normalize scores with balanced approach
        scores = self._normalize_scores_balanced(raw_scores)
        
        # Find primary noise type
        primary_type = max(scores, key=scores.get)
        confidence = scores[primary_type]
        
        # Calculate noise characteristics
        characteristics = self._analyze_noise_characteristics(gray, primary_type)
        
        return {
            'primary_type': primary_type,
            'primary_noise_type': primary_type,
            'confidence': confidence,
            'all_scores': scores,
            'noise_level': characteristics['noise_level'],
            'characteristics': characteristics
        }
    
    def _detect_gaussian_balanced(self, gray):
        """Balanced Gaussian noise detection"""
        
        # Residual analysis
        smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residuals = gray.astype(np.float32) - smooth.astype(np.float32)
        
        # Normality test
        try:
            _, p_value = stats.normaltest(residuals.flatten())
            normality_score = max(0, 1.0 - p_value * 10)
        except:
            normality_score = 0
        
        # Variance consistency
        regions = []
        step = 20
        for i in range(0, gray.shape[0]-step, step):
            for j in range(0, gray.shape[1]-step, step):
                region = residuals[i:i+step, j:j+step]
                if region.size > 0:
                    regions.append(np.var(region))
        
        if len(regions) > 3:
            variance_consistency = 1.0 / (1 + np.std(regions) / (np.mean(regions) + 1e-8))
        else:
            variance_consistency = 0.5
        
        # Extreme pixel penalty
        extreme_pixels = np.sum((gray == 0) | (gray == 255)) / gray.size
        extreme_penalty = max(0, 1.0 - extreme_pixels * 25)
        
        # Noise level check
        noise_std = np.std(residuals)
        noise_score = min(noise_std / 15, 1.0) if noise_std > 5 else 0
        
        # Balanced combination
        gaussian_score = (normality_score * 0.3 + variance_consistency * 0.25 + 
                         extreme_penalty * 0.25 + noise_score * 0.2)
        
        return gaussian_score
    
    def _detect_salt_pepper_balanced(self, gray):
        """Balanced salt & pepper detection"""
        
        # Extreme pixel analysis
        extreme_pixels = np.sum((gray == 0) | (gray == 255))
        total_pixels = gray.size
        extreme_ratio = extreme_pixels / total_pixels
        
        # Balanced response
        if extreme_ratio > 0.15:
            extreme_score = min(extreme_ratio * 3, 0.8)
        elif extreme_ratio > 0.05:
            extreme_score = extreme_ratio * 4
        else:
            extreme_score = extreme_ratio * 2
        
        # Isolation test
        median_filtered = cv2.medianBlur(gray, 3)
        difference = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
        isolated_pixels = np.sum(difference > 100) / total_pixels
        
        # Background stability
        non_extreme_mask = (gray != 0) & (gray != 255)
        if np.sum(non_extreme_mask) > 50:
            non_extreme_pixels = gray[non_extreme_mask]
            background_stability = 1.0 / (1 + np.std(non_extreme_pixels) / 30)
        else:
            background_stability = 0
        
        # Balanced combination
        sp_score = (extreme_score * 0.6 + isolated_pixels * 3 + background_stability * 0.4)
        return min(sp_score, 1.0)
    
    def _detect_speckle_balanced(self, gray):
        """Conservative speckle detection"""
        
        # Multiplicative test (stricter)
        mean_intensity = np.mean(gray)
        if mean_intensity > 20:
            normalized = gray.astype(np.float32) / mean_intensity
            coefficient_of_variation = np.std(normalized) / np.mean(normalized)
            if coefficient_of_variation > 0.2:  # Reasonable threshold
                multiplicative_score = min((coefficient_of_variation - 0.2) * 4, 1.0)
            else:
                multiplicative_score = 0
        else:
            multiplicative_score = 0
        
        # Texture analysis
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        texture_std = np.std(texture_magnitude)
        if texture_std > 25:
            texture_score = min((texture_std - 25) / 25, 1.0)
        else:
            texture_score = 0
        
        # Distribution skewness
        skewness = stats.skew(gray.flatten())
        if skewness > 0.3:
            skew_score = min((skewness - 0.3) / 1.0, 1.0)
        else:
            skew_score = 0
        
        # Combine scores
        speckle_score = (multiplicative_score * 0.5 + texture_score * 0.3 + skew_score * 0.2)
        return min(speckle_score, 1.0)
    
    def _detect_uniform_balanced(self, gray):
        """Balanced uniform noise detection"""
        
        hist, _ = np.histogram(gray, bins=25, density=True)
        hist_variance = np.var(hist)
        hist_mean = np.mean(hist)
        
        if hist_mean > 0:
            flatness_ratio = hist_variance / (hist_mean**2)
            flatness_score = max(0, 1.0 - flatness_ratio * 8) if flatness_ratio < 0.15 else 0
        else:
            flatness_score = 0
        
        gray_range = np.max(gray) - np.min(gray)
        range_score = min(gray_range / 180.0, 1.0) if gray_range > 80 else 0
        
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        structure_penalty = max(0, 1.0 - edge_density * 15)
        
        uniform_score = (flatness_score * 0.5 + range_score * 0.25 + structure_penalty * 0.25)
        return min(uniform_score * 0.6, 1.0)
    
    def _detect_poisson_balanced(self, gray):
        """Balanced Poisson noise detection"""
        
        region_scores = []
        step = 15
        
        for i in range(0, gray.shape[0]-step, step):
            for j in range(0, gray.shape[1]-step, step):
                region = gray[i:i+step, j:j+step]
                mean_val = np.mean(region)
                var_val = np.var(region)
                
                if mean_val > 15:
                    ratio = var_val / mean_val
                    if abs(ratio - 1) < 0.3:
                        score = 1.0 / (1 + abs(ratio - 1) * 5)
                        region_scores.append(score)
        
        if len(region_scores) > 5:
            poisson_score = np.mean(region_scores)
        else:
            poisson_score = 0
        
        return min(poisson_score, 1.0)
    
    def _normalize_scores_balanced(self, scores):
        """Balanced normalization"""
        
        score_values = np.array(list(scores.values()))
        
        # Gentle softmax
        temperature = 2.5
        exp_scores = np.exp(score_values / temperature)
        softmax_scores = exp_scores / np.sum(exp_scores)
        
        # Blend with original scores
        alpha = 0.4
        blended_scores = alpha * softmax_scores + (1 - alpha) * score_values
        
        # Reasonable range
        min_score = 0.25
        max_score = 0.85
        final_scores = np.clip(blended_scores, min_score, max_score)
        
        # Preserve variation
        score_range = np.max(final_scores) - np.min(final_scores)
        if score_range > 0.1:
            final_scores = (final_scores - np.min(final_scores)) / score_range
            final_scores = final_scores * 0.5 + 0.3
        
        normalized_scores = {}
        for i, noise_type in enumerate(self.noise_types):
            normalized_scores[noise_type] = float(final_scores[i])
        
        return normalized_scores
    
    def _analyze_noise_characteristics(self, gray, primary_type):
        """Analyze noise characteristics for the detected type"""
        characteristics = {}
        
        denoised = cv2.medianBlur(gray, 5)
        noise_estimate = gray.astype(np.float32) - denoised.astype(np.float32)
        noise_level = np.std(noise_estimate)
        
        characteristics['noise_level'] = float(noise_level)
        
        if primary_type == 'gaussian':
            characteristics['sigma_estimate'] = float(noise_level)
        elif primary_type == 'salt_pepper':
            extreme_pixels = np.sum((gray == 0) | (gray == 255))
            characteristics['corruption_ratio'] = float(extreme_pixels / gray.size)
        elif primary_type == 'speckle':
            mean_intensity = np.mean(gray)
            characteristics['coefficient_of_variation'] = float(np.std(gray) / (mean_intensity + 1e-8))
        elif primary_type == 'uniform':
            characteristics['range_estimate'] = float(np.max(gray) - np.min(gray))
        elif primary_type == 'poisson':
            characteristics['lambda_estimate'] = float(np.mean(gray))
        
        return characteristics
    
    def estimate_noise_level(self, image):
        """Estimate noise level using robust MAD estimator"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
        return min(sigma / 100.0, 1.0)

    def detect_noise_type(self, image):
        """Alternative method name for compatibility"""
        result = self.detect_noise(image)
        return {
            'primary_noise_type': result['primary_noise_type'],
            'confidence': result['confidence'],
            'all_scores': result['all_scores'],
            'features': {}
        }
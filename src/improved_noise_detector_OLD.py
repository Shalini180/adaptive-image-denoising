# Truly Independent Detector - Each Method Works Independently
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
        """Independent noise detection - each method works on its own merits"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate detection scores independently - no artificial balancing
        raw_scores = {}
        
        # INDEPENDENT DETECTION - each method scores based on its own criteria only
        raw_scores['gaussian'] = self._detect_gaussian_independent(gray)
        raw_scores['salt_pepper'] = self._detect_salt_pepper_independent(gray)
        raw_scores['speckle'] = self._detect_speckle_independent(gray)
        raw_scores['uniform'] = self._detect_uniform_independent(gray)
        raw_scores['poisson'] = self._detect_poisson_independent(gray)
        
        # Natural competition - minimal normalization
        scores = self._normalize_naturally(raw_scores)
        
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
    
    def _detect_gaussian_independent(self, gray):
        """Independent Gaussian detection - scores based on Gaussian characteristics only"""
        
        # 1. Residual analysis for additive noise
        smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
        residuals = gray.astype(np.float32) - smooth.astype(np.float32)
        
        # 2. Statistical normality of residuals
        try:
            _, p_value = stats.normaltest(residuals.flatten())
            if p_value > 0.05:  # Likely normal
                normality_score = min(p_value * 8, 1.0)
            else:
                normality_score = p_value * 4  # Partial credit
        except:
            normality_score = 0.2
        
        # 3. Additive noise pattern (constant variance)
        regions = []
        step = 20
        for i in range(0, gray.shape[0]-step, step):
            for j in range(0, gray.shape[1]-step, step):
                region = residuals[i:i+step, j:j+step]
                if region.size > 50:
                    regions.append(np.var(region))
        
        if len(regions) > 2:
            variance_consistency = 1.0 / (1 + np.std(regions) / (np.mean(regions) + 1e-8))
            if variance_consistency > 0.7:  # Good consistency
                consistency_score = variance_consistency
            else:
                consistency_score = variance_consistency * 0.7  # Partial credit
        else:
            consistency_score = 0.3
        
        # 4. Appropriate noise magnitude
        noise_std = np.std(residuals)
        if 5 < noise_std < 50:
            noise_score = 0.8
        elif 2 < noise_std < 70:
            noise_score = 0.5
        else:
            noise_score = 0.2
        
        # 5. Zero-mean noise (Gaussian should be zero-mean)
        noise_mean = abs(np.mean(residuals))
        if noise_mean < 2:
            mean_score = 0.9
        elif noise_mean < 5:
            mean_score = 0.6
        else:
            mean_score = 0.3
        
        # Independent combination - no artificial restrictions
        gaussian_score = (normality_score * 0.35 + consistency_score * 0.25 + 
                         noise_score * 0.25 + mean_score * 0.15)
        
        return min(gaussian_score, 1.0)
    
    def _detect_salt_pepper_independent(self, gray):
        """Independent salt & pepper detection"""
        
        # 1. Extreme pixel analysis
        extreme_pixels = np.sum((gray == 0) | (gray == 255))
        total_pixels = gray.size
        extreme_ratio = extreme_pixels / total_pixels
        
        if extreme_ratio > 0.05:  # Clear signal
            extreme_score = min(extreme_ratio * 8, 1.0)
        elif extreme_ratio > 0.01:  # Moderate signal
            extreme_score = extreme_ratio * 15
        else:
            extreme_score = extreme_ratio * 5
        
        # 2. Impulse pattern (isolated pixels)
        median_filtered = cv2.medianBlur(gray, 3)
        difference = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
        isolated_pixels = np.sum(difference > 100) / total_pixels
        
        if isolated_pixels > 0.02:
            isolation_score = min(isolated_pixels * 20, 1.0)
        else:
            isolation_score = isolated_pixels * 10
        
        # 3. Background preservation
        non_extreme_mask = (gray != 0) & (gray != 255)
        if np.sum(non_extreme_mask) > 50:
            non_extreme_pixels = gray[non_extreme_mask]
            background_std = np.std(non_extreme_pixels)
            if background_std < 30:  # Stable background
                background_score = 1.0 - (background_std / 50)
            else:
                background_score = 0.3
        else:
            background_score = 0.1
        
        # Independent combination
        sp_score = (extreme_score * 0.5 + isolation_score * 0.3 + background_score * 0.2)
        return min(sp_score, 1.0)
    
    def _detect_speckle_independent(self, gray):
        """Independent speckle detection - no fail-early approach"""
        
        # 1. Multiplicative noise pattern
        mean_intensity = np.mean(gray)
        if mean_intensity > 10:
            normalized = gray.astype(np.float32) / mean_intensity
            coefficient_of_variation = np.std(normalized) / np.mean(normalized)
            if coefficient_of_variation > 0.15:
                multiplicative_score = min(coefficient_of_variation * 3, 1.0)
            else:
                multiplicative_score = coefficient_of_variation * 5  # Partial credit
        else:
            multiplicative_score = 0.1
        
        # 2. Granular texture pattern
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        texture_std = np.std(texture_magnitude)
        
        if texture_std > 15:
            texture_score = min(texture_std / 30, 1.0)
        else:
            texture_score = texture_std / 40  # Partial credit
        
        # 3. Signal-dependent noise (variance proportional to intensity)
        local_means = []
        local_vars = []
        step = 16
        for i in range(0, gray.shape[0]-step, step):
            for j in range(0, gray.shape[1]-step, step):
                region = gray[i:i+step, j:j+step]
                local_means.append(np.mean(region))
                local_vars.append(np.var(region))
        
        if len(local_means) > 3:
            correlation = np.corrcoef(local_means, local_vars)[0,1]
            if correlation > 0.3:  # Positive correlation
                correlation_score = min(correlation * 2, 1.0)
            else:
                correlation_score = max(correlation, 0) / 2  # Partial credit
        else:
            correlation_score = 0.3
        
        # 4. Distribution characteristics
        skewness = stats.skew(gray.flatten())
        if skewness > 0.1:
            skew_score = min(skewness / 1.5, 1.0)
        else:
            skew_score = max(skewness, 0) / 2
        
        # Independent combination - no harsh requirements
        speckle_score = (multiplicative_score * 0.35 + texture_score * 0.25 + 
                        correlation_score * 0.25 + skew_score * 0.15)
        
        return min(speckle_score, 1.0)
    
    def _detect_uniform_independent(self, gray):
        """Independent uniform detection"""
        
        # 1. Histogram uniformity
        hist, _ = np.histogram(gray, bins=20, density=True)
        hist_variance = np.var(hist)
        hist_mean = np.mean(hist)
        
        if hist_mean > 0:
            flatness_ratio = hist_variance / (hist_mean**2)
            if flatness_ratio < 0.3:
                flatness_score = 1.0 - (flatness_ratio / 0.5)
            else:
                flatness_score = 0.2
        else:
            flatness_score = 0.1
        
        # 2. Wide value range
        gray_range = np.max(gray) - np.min(gray)
        if gray_range > 100:
            range_score = min(gray_range / 150, 1.0)
        else:
            range_score = gray_range / 200
        
        # 3. Low spatial structure
        edges = cv2.Canny(gray, 30, 90)
        edge_density = np.sum(edges > 0) / edges.size
        if edge_density < 0.05:
            structure_score = 1.0 - (edge_density / 0.1)
        else:
            structure_score = max(0, 1.0 - edge_density / 0.05)
        
        # 4. Spatial uniformity (low local variance)
        kernel = np.ones((7,7), np.float32) / 49
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        avg_local_var = np.mean(local_variance)
        
        if avg_local_var < 200:
            spatial_score = 1.0 - (avg_local_var / 400)
        else:
            spatial_score = 0.2
        
        # Independent combination
        uniform_score = (flatness_score * 0.3 + range_score * 0.25 + 
                        structure_score * 0.25 + spatial_score * 0.2)
        
        return min(uniform_score, 1.0)
    
    def _detect_poisson_independent(self, gray):
        """Independent Poisson detection - no artificial boost"""
        
        # 1. Variance-to-mean ratio analysis
        region_scores = []
        step = 12
        
        for i in range(0, gray.shape[0]-step, step):
            for j in range(0, gray.shape[1]-step, step):
                region = gray[i:i+step, j:j+step]
                mean_val = np.mean(region)
                var_val = np.var(region)
                
                if mean_val > 10:
                    ratio = var_val / mean_val
                    if abs(ratio - 1) < 0.4:  # Close to 1 (Poisson property)
                        score = 1.0 / (1 + abs(ratio - 1) * 2)
                        region_scores.append(score)
        
        if len(region_scores) > 2:
            local_poisson_score = np.mean(region_scores)
        else:
            local_poisson_score = 0.2
        
        # 2. Global variance-to-mean ratio
        global_mean = np.mean(gray)
        global_var = np.var(gray)
        if global_mean > 15:
            global_ratio = global_var / global_mean
            if abs(global_ratio - 1) < 0.3:
                global_score = 1.0 / (1 + abs(global_ratio - 1) * 3)
            else:
                global_score = 0.3
        else:
            global_score = 0.2
        
        # 3. Intensity level appropriateness
        if 30 < global_mean < 180:
            intensity_score = 0.8
        elif 15 < global_mean < 220:
            intensity_score = 0.5
        else:
            intensity_score = 0.3
        
        # 4. Noise characteristics
        denoised = cv2.medianBlur(gray, 3)
        noise_estimate = gray.astype(np.float32) - denoised.astype(np.float32)
        noise_std = np.std(noise_estimate)
        
        # For Poisson, noise should increase with intensity
        if noise_std > 5:
            expected_noise = np.sqrt(global_mean)  # Theoretical Poisson noise
            noise_ratio = noise_std / (expected_noise + 1)
            if 0.5 < noise_ratio < 2.0:
                noise_score = 1.0 / (1 + abs(noise_ratio - 1))
            else:
                noise_score = 0.3
        else:
            noise_score = 0.2
        
        # Independent combination - no artificial boost
        poisson_score = (local_poisson_score * 0.35 + global_score * 0.25 + 
                        intensity_score * 0.2 + noise_score * 0.2)
        
        return min(poisson_score, 1.0)
    
    def _normalize_naturally(self, scores):
        """Natural normalization - preserves true differences"""
        
        score_values = np.array(list(scores.values()))
        
        # Extremely minimal processing to preserve natural competition
        
        # Step 1: Tiny enhancement of natural differences
        score_mean = np.mean(score_values)
        enhanced_scores = score_values + (score_values - score_mean) * 0.01  # Minimal
        
        # Step 2: Ultra-gentle softmax (almost no effect)
        temperature = 15.0  # Very high = almost no normalization
        exp_scores = np.exp(enhanced_scores / temperature)
        softmax_scores = exp_scores / np.sum(exp_scores)
        
        # Step 3: Preserve 95% of original scores
        alpha = 0.05  # Almost no softmax influence
        blended_scores = alpha * softmax_scores + (1 - alpha) * enhanced_scores
        
        # Step 4: Wide natural range
        min_score = 0.05
        max_score = 1.0
        final_scores = np.clip(blended_scores, min_score, max_score)
        
        # Step 5: No normalization unless scores are identical
        score_range = np.max(final_scores) - np.min(final_scores)
        if score_range < 0.01:  # Only if virtually identical
            final_scores = (final_scores - np.min(final_scores)) / max(score_range, 0.001)
            final_scores = final_scores * 0.1 + 0.5  # Very narrow range
        
        # Create result dictionary
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
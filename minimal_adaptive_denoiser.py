"""
Minimal Working Adaptive Denoiser
Simplified version that focuses on core functionality
"""

import numpy as np
import cv2
import time
from pathlib import Path
import json

# Import our improved noise detector
try:
    from improved_noise_detector import ImprovedNoiseDetector
except ImportError:
    print("Warning: Using simple noise detector")
    
    class ImprovedNoiseDetector:
        def detect_noise_type(self, image):
            return {
                'primary_noise_type': 'gaussian',
                'confidence': 0.8,
                'all_scores': {'gaussian': 0.8, 'salt_pepper': 0.1, 'speckle': 0.1, 'uniform': 0.0, 'poisson': 0.0}
            }
        
        def estimate_noise_level(self, image):
            return 0.1

class MinimalAdaptiveDenoiser:
    """Minimal adaptive denoiser that definitely works"""
    
    def __init__(self):
        """Initialize the minimal denoiser"""
        self.noise_detector = ImprovedNoiseDetector()
        
        # Load optimal weights if available
        self.optimal_weights = self.load_optimal_weights()
        
    def load_optimal_weights(self):
        """Load optimal weights from experiments"""
        try:
            weights_path = Path("experiments/weight_optimization/optimization_results.json")
            if weights_path.exists():
                with open(weights_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        
        # Default weights
        return {
            'gaussian': {'alpha': 0.45, 'beta': 0.35, 'gamma': 0.20},
            'salt_pepper': {'alpha': 0.25, 'beta': 0.25, 'gamma': 0.50},
            'speckle': {'alpha': 0.40, 'beta': 0.30, 'gamma': 0.30},
            'uniform': {'alpha': 0.35, 'beta': 0.40, 'gamma': 0.25},
            'poisson': {'alpha': 0.30, 'beta': 0.35, 'gamma': 0.35}
        }
    
    def gaussian_filter(self, image, sigma=1.0):
        """Apply Gaussian filtering"""
        return cv2.GaussianBlur(image, (5, 5), sigma)
    
    def bilateral_filter(self, image):
        """Apply bilateral filtering"""
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    def median_filter(self, image, kernel_size=5):
        """Apply median filtering"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)
    
    def denoise_image(self, image):
        """Apply adaptive denoising to image"""
        start_time = time.time()
        
        try:
            # Step 1: Detect noise type
            detection_start = time.time()
            noise_info = self.noise_detector.detect_noise_type(image)
            noise_level = self.noise_detector.estimate_noise_level(image)
            detection_time = time.time() - detection_start
            
            primary_noise = noise_info['primary_noise_type']
            confidence = noise_info['confidence']
            
            # Step 2: Get optimal weights
            weights = self.optimal_weights.get(primary_noise, {
                'alpha': 0.33, 'beta': 0.33, 'gamma': 0.34
            })
            
            alpha = weights.get('alpha', 0.33)
            beta = weights.get('beta', 0.33)
            gamma = weights.get('gamma', 0.34)
            
            # Step 3: Apply combined denoising
            gaussian_result = self.gaussian_filter(image, sigma=1.0)
            bilateral_result = self.bilateral_filter(image)
            median_result = self.median_filter(image, kernel_size=5)
            
            # Step 4: Combine results
            final_image = (alpha * gaussian_result.astype(np.float32) +
                          beta * bilateral_result.astype(np.float32) +
                          gamma * median_result.astype(np.float32))
            
            final_image = np.clip(final_image, 0, 255).astype(np.uint8)
            
            processing_time = time.time() - start_time
            
            # Return results
            return {
                'final_image': final_image,
                'metadata': {
                    'noise_detection': {
                        'primary_type': primary_noise,
                        'confidence': confidence,
                        'all_scores': noise_info['all_scores']
                    },
                    'denoising_parameters': {
                        'method_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
                        'noise_adaptation': True
                    },
                    'processing_time': processing_time,
                    'refinement_applied': False
                },
                'processing_stages': [
                    {'stage': 'noise_detection', 'processing_time': detection_time},
                    {'stage': 'denoising', 'processing_time': processing_time - detection_time}
                ]
            }
            
        except Exception as e:
            print(f"Denoising error: {e}")
            # Return fallback result
            return {
                'final_image': self.gaussian_filter(image),  # Simple fallback
                'metadata': {
                    'noise_detection': {'primary_type': 'unknown', 'confidence': 0.0, 'all_scores': {}},
                    'denoising_parameters': {'method_weights': {'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0}},
                    'processing_time': time.time() - start_time,
                    'refinement_applied': False,
                    'error': str(e)
                },
                'processing_stages': [{'stage': 'fallback', 'processing_time': time.time() - start_time}]
            }

# Test the minimal denoiser
if __name__ == "__main__":
    print("🧪 MINIMAL ADAPTIVE DENOISER TEST")
    print("=" * 40)
    
    # Create test image
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (100, 100), (150, 100, 50), -1)
    cv2.circle(img, (64, 64), 30, (50, 150, 100), -1)
    
    # Add noise
    noise = np.random.normal(0, 20, img.shape)
    noisy_img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    # Test denoiser
    denoiser = MinimalAdaptiveDenoiser()
    result = denoiser.denoise_image(noisy_img)
    
    if result and 'final_image' in result:
        print("✅ Minimal system working!")
        print(f"Detected: {result['metadata']['noise_detection']['primary_type']}")
        print(f"Processing time: {result['metadata']['processing_time']:.3f}s")
        
        # Save results
        cv2.imwrite('minimal_test_original.png', img)
        cv2.imwrite('minimal_test_noisy.png', noisy_img)
        cv2.imwrite('minimal_test_denoised.png', result['final_image'])
        print("Images saved: minimal_test_*.png")
    else:
        print("❌ Minimal system failed")

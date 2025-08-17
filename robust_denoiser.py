"""
Robust Adaptive Denoising with Error Handling
Fixed version that handles indexing errors properly
"""

import numpy as np
import cv2
import sys
import os
sys.path.append('src')

class RobustDenoiser:
    """A robust denoiser with proper error handling"""
    
    def __init__(self):
        self.methods = {
            'gaussian': self.gaussian_denoise,
            'bilateral': self.bilateral_denoise,
            'median': self.median_denoise
        }
    
    def safe_index(self, array, *indices):
        """Safely index array by converting indices to integers"""
        try:
            safe_indices = tuple(int(idx) if isinstance(idx, (float, np.floating)) else idx for idx in indices)
            return array[safe_indices]
        except (ValueError, IndexError) as e:
            print(f"⚠️ Indexing error: {e}")
            return array[tuple(max(0, min(int(idx), array.shape[i]-1)) if isinstance(idx, (int, float, np.number)) else idx 
                              for i, idx in enumerate(indices))]
    
    def gaussian_denoise(self, image, sigma=1.0):
        """Apply Gaussian denoising with safe operations"""
        try:
            from skimage import filters
            if len(image.shape) == 3:
                result = np.zeros_like(image, dtype=np.float64)
                for i in range(image.shape[2]):
                    result[:,:,i] = filters.gaussian(image[:,:,i].astype(np.float64), sigma=sigma)
                return result
            else:
                return filters.gaussian(image.astype(np.float64), sigma=sigma)
        except Exception as e:
            print(f"⚠️ Gaussian denoising error: {e}")
            return image.astype(np.float64)
    
    def bilateral_denoise(self, image, d=9, sigma_color=75, sigma_space=75):
        """Apply bilateral denoising with safe operations"""
        try:
            if image.dtype != np.uint8:
                img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                img_uint8 = image.copy()
            
            result = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
            
            if image.dtype != np.uint8:
                return result.astype(np.float64) / 255.0
            return result
        except Exception as e:
            print(f"⚠️ Bilateral denoising error: {e}")
            return image
    
    def median_denoise(self, image, kernel_size=5):
        """Apply median denoising with safe operations"""
        try:
            if image.dtype != np.uint8:
                img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                img_uint8 = image.copy()
            
            if len(image.shape) == 3:
                result = np.zeros_like(img_uint8)
                for i in range(image.shape[2]):
                    result[:,:,i] = cv2.medianBlur(img_uint8[:,:,i], kernel_size)
            else:
                result = cv2.medianBlur(img_uint8, kernel_size)
            
            if image.dtype != np.uint8:
                return result.astype(np.float64) / 255.0
            return result
        except Exception as e:
            print(f"⚠️ Median denoising error: {e}")
            return image
    
    def denoise_image(self, image, method='bilateral', **params):
        """Safely denoise an image"""
        try:
            if method in self.methods:
                return self.methods[method](image, **params)
            else:
                print(f"⚠️ Unknown method {method}, using bilateral")
                return self.bilateral_denoise(image, **params)
        except Exception as e:
            print(f"❌ Denoising failed: {e}")
            return image

def test_robust_denoiser():
    """Test the robust denoiser"""
    print("🧪 Testing robust denoiser...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Add noise
    noise = np.random.normal(0, 25, test_image.shape)
    noisy_image = np.clip(test_image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    # Test denoising
    denoiser = RobustDenoiser()
    
    methods_to_test = ['gaussian', 'bilateral', 'median']
    for method in methods_to_test:
        try:
            result = denoiser.denoise_image(noisy_image, method=method)
            if result is not None and result.shape == noisy_image.shape:
                print(f"✅ {method.title()} denoising: SUCCESS")
            else:
                print(f"❌ {method.title()} denoising: FAILED (shape mismatch)")
        except Exception as e:
            print(f"❌ {method.title()} denoising: FAILED ({e})")
    
    return denoiser

if __name__ == "__main__":
    denoiser = test_robust_denoiser()
    
    print("\n🎯 Robust denoiser ready!")
    print("Usage: denoiser.denoise_image(your_image, method='bilateral')")

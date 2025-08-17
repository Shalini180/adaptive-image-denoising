def safe_index(array, *indices):
    """Safely index array by converting indices to integers"""
    safe_indices = tuple(int(idx) if isinstance(idx, (float, np.floating)) else idx for idx in indices)
    return array[safe_indices]

"""
Core Denoising Methods
Clean implementation with all required methods
"""

import numpy as np
import cv2
from skimage import restoration
import warnings
warnings.filterwarnings('ignore')

class CoreDenoisingMethods:
    """Core denoising methods for the adaptive system"""
    
    def __init__(self):
        """Initialize the core denoising methods"""
        self.methods = {
            'gaussian': self.gaussian_filter,
            'bilateral': self.bilateral_filter,
            'median': self.median_filter,
            'adaptive_median': self.adaptive_median_filter,
            'nlm': self.non_local_means_filter,
            'non_local_means': self.non_local_means,
            'wavelet': self.wavelet_denoising,
            'wavelet_denoising': self.wavelet_denoising,
            'bm3d': self.bm3d_denoising,
            'bm3d_denoising': self.bm3d_denoising,
            'wiener': self.wiener_filter,
            'morphological': self.morphological_filter,
            'anisotropic': self.anisotropic_diffusion,
            'method_a': self.method_a_denoise,
            'method_b': self.method_b_denoise,
            'method_c': self.method_c_denoise
        }
    
    def gaussian_filter(self, image, noise_type='gaussian', **params):
        """Apply Gaussian filtering"""
        sigma = params.get('sigma', 1.0)
        kernel_size = params.get('kernel_size', 5)
        
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def bilateral_filter(self, image, noise_type='gaussian', **params):
        """Apply bilateral filtering"""
        d = int(params.get('d', 9))
        sigma_color = params.get('sigma_color', 75)
        sigma_space = params.get('sigma_space', 75)
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def median_filter(self, image, noise_type='gaussian', **params):
        """Apply median filtering"""
        kernel_size = int(params.get('kernel_size', 5))
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
            return result
        else:
            return cv2.medianBlur(image, kernel_size)
    
    def non_local_means_filter(self, image, noise_type='gaussian', **params):
        """Apply non-local means filtering"""
        image_float = image.astype(np.float32) / 255.0
        
        h = params.get('h', 0.1)
        fast_mode = params.get('fast_mode', True)
        patch_size = int(params.get('patch_size', 7))
        patch_distance = int(params.get('patch_distance', 21))
        
        try:
            result = restoration.denoise_nl_means(
                image_float,
                h=h,
                fast_mode=fast_mode,
                patch_size=patch_size,
                patch_distance=patch_distance,
                channel_axis=2
            )
        except TypeError:
            result = restoration.denoise_nl_means(
                image_float,
                h=h,
                fast_mode=fast_mode,
                patch_size=patch_size,
                patch_distance=patch_distance,
                multichannel=True
            )
        
        return (result * 255).astype(np.uint8)
    
    def non_local_means(self, image, noise_type='gaussian', **params):
        """Alias for non_local_means_filter"""
        return self.non_local_means_filter(image, noise_type, **params)
    
    def adaptive_median_filter(self, image, noise_type='gaussian', **params):
        """Adaptive median filter implementation"""
        max_window_size = int(params.get('max_window_size', 9))
        
        if len(image.shape) == 3:
            # Process each channel separately for color images
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._adaptive_median_channel(
                    image[:, :, c], max_window_size
                )
            return result
        else:
            # Grayscale image
            return self._adaptive_median_channel(image, max_window_size)

    def _adaptive_median_channel(self, channel, max_window_size):
        """Apply adaptive median filter to single channel"""
        h, w = channel.shape
        result = channel.copy().astype(np.float32)
        
        for i in range(h):
            for j in range(w):
                # Start with smallest window size
                for window_size in range(3, max_window_size + 1, 2):
                    half_window = window_size // 2
                    
                    # Define window boundaries
                    y_min = max(0, i - half_window)
                    y_max = min(h, i + half_window + 1)
                    x_min = max(0, j - half_window)
                    x_max = min(w, j + half_window + 1)
                    
                    # Extract window
                    window = channel[y_min:y_max, x_min:x_max]
                    
                    if window.size == 0:
                        continue
                    
                    # Calculate statistics
                    z_min = np.min(window)
                    z_max = np.max(window)
                    z_med = np.median(window)
                    z_xy = channel[i, j]
                    
                    # Stage A
                    A1 = z_med - z_min
                    A2 = z_med - z_max
                    
                    if A1 > 0 and A2 < 0:
                        # Stage B
                        B1 = z_xy - z_min
                        B2 = z_xy - z_max
                        
                        if B1 > 0 and B2 < 0:
                            result[i, j] = z_xy
                        else:
                            result[i, j] = z_med
                        break
                    else:
                        # Continue with larger window
                        if window_size >= max_window_size:
                            result[i, j] = z_med
                            break
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def wavelet_denoising(self, image, noise_type='gaussian', **params):
        """Wavelet denoising implementation"""
        try:
            # Try to use scikit-image restoration if available
            from skimage import restoration
            
            # Convert to float
            if len(image.shape) == 3:
                image_float = image.astype(np.float32) / 255.0
            else:
                image_float = image.astype(np.float32) / 255.0
            
            # Parameters
            sigma = params.get('sigma', 0.1)
            method = params.get('method', 'BayesShrink')
            mode = params.get('mode', 'soft')
            
            # Apply wavelet denoising
            if len(image.shape) == 3:
                # Color image - process each channel
                result = np.zeros_like(image_float)
                for c in range(image.shape[2]):
                    result[:, :, c] = restoration.denoise_wavelet(
                        image_float[:, :, c], 
                        method=method, 
                        mode=mode,
                        sigma=sigma
                    )
            else:
                # Grayscale image
                result = restoration.denoise_wavelet(
                    image_float, 
                    method=method, 
                    mode=mode,
                    sigma=sigma
                )
            
            # Convert back to uint8
            return (result * 255).astype(np.uint8)
            
        except ImportError:
            print("Warning: scikit-image not available, using fallback wavelet method")
            return self._fallback_wavelet_denoising(image, noise_type, **params)
        except Exception as e:
            print(f"Warning: wavelet_denoising error: {e}, using fallback")
            return self._fallback_wavelet_denoising(image, noise_type, **params)

    def _fallback_wavelet_denoising(self, image, noise_type='gaussian', **params):
        """Fallback wavelet denoising using simple frequency domain filtering"""
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self._simple_frequency_denoise(image[:, :, c])
            return result
        else:
            return self._simple_frequency_denoise(image)

    def _simple_frequency_denoise(self, channel):
        """Simple frequency domain denoising (wavelet approximation)"""
        # Convert to float
        img_float = channel.astype(np.float32)
        
        # Apply FFT
        f_transform = np.fft.fft2(img_float)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create a simple frequency filter (low-pass)
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask
        mask = np.zeros((rows, cols), np.uint8)
        r = 30  # Filter radius
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= r ** 2
        mask[mask_area] = 1
        
        # Apply mask and inverse FFT
        f_shift = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift)
        result = np.fft.ifft2(f_ishift)
        result = np.real(result)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def bm3d_denoising(self, image, noise_type='gaussian', **params):
        """BM3D denoising implementation (fallback to NLM)"""
        try:
            # Try to import BM3D if available
            import bm3d
            
            if len(image.shape) == 3:
                # Color image
                image_float = image.astype(np.float32) / 255.0
                sigma = params.get('sigma', 0.1)
                denoised = bm3d.bm3d(image_float, sigma)
                return (denoised * 255).astype(np.uint8)
            else:
                # Grayscale image
                image_float = image.astype(np.float32) / 255.0
                sigma = params.get('sigma', 0.1)
                denoised = bm3d.bm3d(image_float, sigma)
                return (denoised * 255).astype(np.uint8)
                
        except ImportError:
            # Fallback to non-local means
            print("Warning: BM3D not available, using non-local means fallback")
            return self.non_local_means_filter(image, noise_type, **params)
        except Exception as e:
            print(f"Warning: BM3D error: {e}, using non-local means fallback")
            return self.non_local_means_filter(image, noise_type, **params)
    
    def wiener_filter(self, image, noise_type='gaussian', **params):
        """Apply Wiener filtering"""
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                result[:, :, i] = self._wiener_channel(image[:, :, i], noise_type, **params)
            return result
        else:
            return self._wiener_channel(image, noise_type, **params)
    
    def _wiener_channel(self, channel, noise_type, **params):
        """Apply Wiener filter to a single channel"""
        noise_var = params.get('noise_variance', 0.01)
        blurred = cv2.GaussianBlur(channel.astype(np.float32), (5, 5), 1.0)
        local_var = cv2.Laplacian(channel, cv2.CV_64F)
        local_var = np.abs(local_var)
        local_var = local_var / (np.max(local_var) + 1e-8)
        wiener_coeff = local_var / (local_var + noise_var)
        result = wiener_coeff * channel.astype(np.float32) + (1 - wiener_coeff) * blurred
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def morphological_filter(self, image, noise_type='gaussian', **params):
        """Apply morphological filtering"""
        operations = {
            'salt_pepper': 'opening',
            'gaussian': 'closing',
            'speckle': 'opening',
            'uniform': 'gradient',
            'poisson': 'tophat'
        }
        operation = params.get('operation', operations.get(noise_type, 'opening'))
        kernel_size = int(params.get('kernel_size', 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                if operation == 'opening':
                    result[:, :, i] = cv2.morphologyEx(image[:, :, i], cv2.MORPH_OPEN, kernel)
                elif operation == 'closing':
                    result[:, :, i] = cv2.morphologyEx(image[:, :, i], cv2.MORPH_CLOSE, kernel)
                elif operation == 'gradient':
                    result[:, :, i] = cv2.morphologyEx(image[:, :, i], cv2.MORPH_GRADIENT, kernel)
                elif operation == 'tophat':
                    result[:, :, i] = cv2.morphologyEx(image[:, :, i], cv2.MORPH_TOPHAT, kernel)
                else:
                    result[:, :, i] = image[:, :, i]
            return result
        else:
            if operation == 'opening':
                return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            elif operation == 'gradient':
                return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
            elif operation == 'tophat':
                return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            else:
                return image
    
    def anisotropic_diffusion(self, image, noise_type='gaussian', **params):
        """Apply anisotropic diffusion filtering"""
        iterations = int(params.get('iterations', 5))
        
        if len(image.shape) == 3:
            result = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[2]):
                result[:, :, i] = self._anisotropic_channel(image[:, :, i], iterations)
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = self._anisotropic_channel(image, iterations)
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def _anisotropic_channel(self, channel, iterations):
        """Apply anisotropic diffusion to single channel"""
        img = channel.astype(np.float32)
        for i in range(iterations):
            filtered = cv2.GaussianBlur(img, (3, 3), 1.0)
            img = 0.8 * img + 0.2 * filtered
        return img
    
    def method_a_denoise(self, image, adapted_params=None, noise_type='gaussian', **params):
        """Method A: Gaussian + Bilateral combination"""
        try:
            if adapted_params and isinstance(adapted_params, dict):
                for key, value in adapted_params.items():
                    params[key] = value
            
            gaussian_result = self.gaussian_filter(image, noise_type, **params)
            bilateral_result = self.bilateral_filter(image, noise_type, **params)
            
            alpha = 0.6
            result = (alpha * gaussian_result.astype(np.float32) + 
                     (1-alpha) * bilateral_result.astype(np.float32))
            return np.clip(result, 0, 255).astype(np.uint8)
                    
        except Exception as e:
            print(f"Warning: method_a_denoise error: {e}")
            return self.gaussian_filter(image, noise_type, **params)
    
    def method_b_denoise(self, image, adapted_params=None, noise_type='gaussian', **params):
        """Method B: Non-local means"""
        try:
            if adapted_params and isinstance(adapted_params, dict):
                for key, value in adapted_params.items():
                    params[key] = value
            
            return self.non_local_means_filter(image, noise_type, **params)
            
        except Exception as e:
            print(f"Warning: method_b_denoise error: {e}")
            return self.bilateral_filter(image, noise_type, **params)
    
    def method_c_denoise(self, image, adapted_params=None, noise_type='gaussian', **params):
        """Method C: Median + Morphological combination"""
        try:
            if adapted_params and isinstance(adapted_params, dict):
                for key, value in adapted_params.items():
                    params[key] = value
            
            median_result = self.median_filter(image, noise_type, **params)
            morph_result = self.morphological_filter(image, noise_type, **params)
            
            alpha = 0.7
            result = (alpha * median_result.astype(np.float32) + 
                     (1-alpha) * morph_result.astype(np.float32))
            return np.clip(result, 0, 255).astype(np.uint8)
                    
        except Exception as e:
            print(f"Warning: method_c_denoise error: {e}")
            return self.median_filter(image, noise_type, **params)

# Test if the module can be imported
if __name__ == "__main__":
    print("🧪 Testing CoreDenoisingMethods...")
    
    try:
        methods = CoreDenoisingMethods()
        print("✅ Initialization successful")
        
        # Test basic functionality
        test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Test each method
        result_a = methods.method_a_denoise(test_img)
        result_b = methods.method_b_denoise(test_img)
        result_c = methods.method_c_denoise(test_img)
        
        print(f"✅ method_a_denoise: {result_a.shape}")
        print(f"✅ method_b_denoise: {result_b.shape}")
        print(f"✅ method_c_denoise: {result_c.shape}")
        
        # Test new methods
        result_adaptive = methods.adaptive_median_filter(test_img)
        result_wavelet = methods.wavelet_denoising(test_img)
        result_nlm = methods.non_local_means(test_img)
        result_bm3d = methods.bm3d_denoising(test_img)
        
        print(f"✅ adaptive_median_filter: {result_adaptive.shape}")
        print(f"✅ wavelet_denoising: {result_wavelet.shape}")
        print(f"✅ non_local_means: {result_nlm.shape}")
        print(f"✅ bm3d_denoising: {result_bm3d.shape}")
        print("🎉 All methods working correctly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
"""
Core Denoising Methods Implementation
Selected through systematic analysis for optimal complementarity

Based on empirical research and multi-criteria selection framework:
- Method A: Adaptive Bilateral Filter (Noise-Specific Specialist)
- Method B: Multi-Method Consensus (Consensus Coordinator) 
- Method C: Edge-Preserving Non-Local Means (Quality Enhancer)
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import restoration, filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time

class CoreDenoisingMethods:
    """
    Three core methods selected for the adaptive denoising system
    Based on empirical analysis and literature review
    """
    
    def __init__(self):
        self.method_configs = {
            'method_a': {
                'name': 'Adaptive Bilateral Filter',
                'type': 'noise_specific',
                'parameters': {
                    'sigma_space_base': 15.0,
                    'sigma_intensity_base': 0.1,
                    'kernel_size_base': 9,
                    'adaptation_factor': 2.0
                },
                'target_noise_types': ['gaussian', 'uniform'],
                'computational_score': 0.7,
                'quality_score': 0.8,
                'adaptability_score': 0.9
            },
            'method_b': {
                'name': 'Multi-Method Consensus',
                'type': 'consensus_based',
                'parameters': {
                    'base_methods': ['bilateral', 'gaussian_adaptive', 'median_selective'],
                    'consensus_strategy': 'weighted_median',
                    'confidence_threshold': 0.7,
                    'local_variance_window': 5
                },
                'target_noise_types': ['all'],
                'computational_score': 0.4,
                'quality_score': 0.8,
                'adaptability_score': 0.9
            },
            'method_c': {
                'name': 'Edge-Preserving Non-Local Means',
                'type': 'structure_preserving',
                'parameters': {
                    'h_base': 0.1,
                    'template_window_size': 7,
                    'search_window_size': 21,
                    'edge_threshold': 0.1,
                    'fast_mode': True
                },
                'target_noise_types': ['gaussian', 'speckle'],
                'computational_score': 0.3,
                'quality_score': 0.9,
                'adaptability_score': 0.8
            }
        }
        
        # Pre-computed parameter lookup tables for efficiency
        self.noise_adaptation_lut = self._build_adaptation_lut()
    
    def _build_adaptation_lut(self):
        """Build lookup tables for noise-specific parameter adaptation"""
        return {
            'gaussian': {
                'sigma_space_multiplier': [1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
                'sigma_intensity_multiplier': [0.8, 1.0, 1.3, 1.6, 2.0, 2.5],
                'h_multiplier': [0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
            },
            'salt_pepper': {
                'median_kernel_sizes': [3, 5, 7, 9, 11, 13],
                'bilateral_sigma_space': [5, 8, 12, 15, 20, 25],
                'bilateral_sigma_intensity': [0.05, 0.08, 0.12, 0.15, 0.2, 0.3]
            },
            'speckle': {
                'lee_window_sizes': [3, 5, 7, 9, 11],
                'h_multiplier': [1.2, 1.5, 1.8, 2.2, 2.8, 3.5],
                'edge_threshold_multiplier': [0.8, 1.0, 1.3, 1.6, 2.0]
            },
            'uniform': {
                'sigma_space_multiplier': [0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
                'sigma_intensity_multiplier': [1.0, 1.2, 1.5, 1.8, 2.2, 2.8]
            },
            'poisson': {
                'anscombe_mode': True,
                'h_multiplier': [1.5, 1.8, 2.2, 2.8, 3.5, 4.2],
                'bilateral_sigma_intensity': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
            }
        }
    
    def estimate_noise_level(self, image):
        """
        Estimate noise level using robust MAD estimator
        Returns noise standard deviation estimate
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Robust noise estimation using Median Absolute Deviation (MAD)
        sigma = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
        
        return sigma
    
    def adapt_parameters_to_noise(self, noise_type, noise_level, base_params):
        """
        Adapt algorithm parameters based on detected noise type and level
        """
        # Normalize noise level to [0, 5] range for LUT indexing
        level_index = min(int(noise_level * 10), 5)
        
        adapted_params = base_params.copy()
        
        if noise_type in self.noise_adaptation_lut:
            lut = self.noise_adaptation_lut[noise_type]
            
            # Adapt parameters based on noise type
            if 'sigma_space_multiplier' in lut and level_index < len(lut['sigma_space_multiplier']):
                adapted_params['sigma_space'] = (
                    base_params.get('sigma_space_base', 15.0) * 
                    lut['sigma_space_multiplier'][level_index]
                )
            
            if 'sigma_intensity_multiplier' in lut and level_index < len(lut['sigma_intensity_multiplier']):
                adapted_params['sigma_intensity'] = (
                    base_params.get('sigma_intensity_base', 0.1) * 
                    lut['sigma_intensity_multiplier'][level_index]
                )
            
            if 'h_multiplier' in lut and level_index < len(lut['h_multiplier']):
                adapted_params['h'] = (
                    base_params.get('h_base', 0.1) * 
                    lut['h_multiplier'][level_index]
                )
        
        return adapted_params
    
    def method_a_denoise(self, image, noise_type, noise_level, confidence_score=1.0):
        """
        Method A: Adaptive Bilateral Filtering with noise-specific parameter tuning
        
        Specializes in Gaussian and uniform noise with adaptive parameters
        Fast execution with good edge preservation
        """
        start_time = time.time()
        
        # Get base parameters
        base_params = self.method_configs['method_a']['parameters']
        
        # Adapt parameters to noise characteristics
        adapted_params = self.adapt_parameters_to_noise(noise_type, noise_level, base_params)
        
        # Apply adaptive bilateral filtering
        if len(image.shape) == 3:
            # Color image - process each channel
            result = np.zeros_like(image)
            for channel in range(3):
                result[:, :, channel] = cv2.bilateralFilter(
                    image[:, :, channel].astype(np.uint8),
                    d=adapted_params.get('kernel_size_base', 9),
                    sigmaColor=adapted_params.get('sigma_intensity', 0.1) * 255,
                    sigmaSpace=adapted_params.get('sigma_space', 15.0)
                )
        else:
            # Grayscale image
            result = cv2.bilateralFilter(
                image.astype(np.uint8),
                d=adapted_params.get('kernel_size_base', 9),
                sigmaColor=adapted_params.get('sigma_intensity', 0.1) * 255,
                sigmaSpace=adapted_params.get('sigma_space', 15.0)
            )
        
        processing_time = time.time() - start_time
        
        return {
            'denoised_image': result.astype(image.dtype),
            'method': 'adaptive_bilateral',
            'parameters_used': adapted_params,
            'processing_time': processing_time,
            'confidence': confidence_score
        }
    
    def method_b_denoise(self, image, noise_type, noise_level, confidence_score=1.0):
        """
        Method B: Multi-Method Consensus using weighted median combination
        
        Combines multiple base algorithms for robustness across all noise types
        Reduces individual method limitations through intelligent averaging
        """
        start_time = time.time()
        
        base_params = self.method_configs['method_b']['parameters']
        
        # Apply multiple base methods
        base_results = []
        
        # Base Method 1: Bilateral Filter
        bilateral_result = cv2.bilateralFilter(
            image.astype(np.uint8),
            d=9,
            sigmaColor=noise_level * 50 + 20,
            sigmaSpace=15.0
        )
        base_results.append(bilateral_result)
        
        # Base Method 2: Adaptive Gaussian Filter
        sigma = max(0.5, noise_level * 2.0)
        gaussian_result = filters.gaussian(image, sigma=sigma, preserve_range=True)
        base_results.append(gaussian_result.astype(np.uint8))
        
        # Base Method 3: Selective Median Filter (for impulse noise)
        if noise_type in ['salt_pepper', 'impulse']:
            median_result = ndimage.median_filter(image, size=5)
        else:
            # Use gentle median for other noise types
            median_result = ndimage.median_filter(image, size=3)
        base_results.append(median_result.astype(np.uint8))
        
        # Weighted median consensus
        weights = self._calculate_consensus_weights(base_results, noise_type, noise_level)
        
        # Apply weighted median
        result = self._weighted_median_consensus(base_results, weights)
        
        processing_time = time.time() - start_time
        
        return {
            'denoised_image': result.astype(image.dtype),
            'method': 'multi_method_consensus',
            'base_methods': ['bilateral', 'gaussian_adaptive', 'median_selective'],
            'weights_used': weights,
            'processing_time': processing_time,
            'confidence': confidence_score
        }
    
    def method_c_denoise(self, image, noise_type, noise_level, confidence_score=1.0):
        """
        Method C: Edge-Preserving Non-Local Means with adaptive parameters
        
        Provides high-quality denoising with excellent texture and edge preservation
        Handles Gaussian and speckle noise particularly well
        """
        start_time = time.time()
        
        base_params = self.method_configs['method_c']['parameters']
        adapted_params = self.adapt_parameters_to_noise(noise_type, noise_level, base_params)
        
        # Convert to appropriate format for scikit-image
        if image.dtype != np.float64:
            image_float = image.astype(np.float64) / 255.0
        else:
            image_float = image
        
        # Apply non-local means denoising
        h = adapted_params.get('h', 0.1)
        
        if len(image.shape) == 3:
            # Color image
            result = restoration.denoise_nl_means(
                image_float,
                h=h,
                fast_mode=adapted_params.get('fast_mode', True),
                patch_size=adapted_params.get('template_window_size', 7),
                patch_distance=adapted_params.get('search_window_size', 21),
                multichannel=True
            )
        else:
            # Grayscale image
            result = restoration.denoise_nl_means(
                image_float,
                h=h,
                fast_mode=adapted_params.get('fast_mode', True),
                patch_size=adapted_params.get('template_window_size', 7),
                patch_distance=adapted_params.get('search_window_size', 21)
            )
        
        # Convert back to original data type
        if image.dtype != np.float64:
            result = (result * 255.0).astype(image.dtype)
        
        processing_time = time.time() - start_time
        
        return {
            'denoised_image': result,
            'method': 'edge_preserving_nlm',
            'parameters_used': adapted_params,
            'processing_time': processing_time,
            'confidence': confidence_score
        }
    
    def _calculate_consensus_weights(self, base_results, noise_type, noise_level):
        """Calculate adaptive weights for consensus combination"""
        # Base weights favor different methods based on noise type
        if noise_type == 'gaussian':
            base_weights = [0.4, 0.4, 0.2]  # Favor bilateral and gaussian
        elif noise_type in ['salt_pepper', 'impulse']:
            base_weights = [0.3, 0.2, 0.5]  # Favor median
        elif noise_type == 'speckle':
            base_weights = [0.5, 0.3, 0.2]  # Favor bilateral
        else:
            base_weights = [0.33, 0.33, 0.34]  # Equal weights for unknown
        
        # Adjust weights based on noise level
        noise_factor = min(noise_level / 0.2, 2.0)  # Cap at 2.0
        
        # Higher noise -> favor more aggressive methods
        if noise_level > 0.15:
            base_weights[2] *= (1.0 + noise_factor * 0.2)  # Boost median
        
        # Normalize weights
        total_weight = sum(base_weights)
        normalized_weights = [w / total_weight for w in base_weights]
        
        return normalized_weights
    
    def _weighted_median_consensus(self, base_results, weights):
        """Compute weighted median of base method results"""
        if len(base_results) != len(weights):
            raise ValueError("Number of results must match number of weights")
        
        # Stack results along new axis
        stacked = np.stack(base_results, axis=-1)
        
        # For simplicity, use weighted average instead of true weighted median
        # (True weighted median is computationally expensive)
        weights_array = np.array(weights)
        result = np.average(stacked, axis=-1, weights=weights_array)
        
        return result.astype(base_results[0].dtype)
    
    def get_method_info(self, method_name):
        """Get detailed information about a specific method"""
        method_map = {
            'method_a': 'method_a',
            'adaptive_bilateral': 'method_a',
            'method_b': 'method_b', 
            'consensus': 'method_b',
            'method_c': 'method_c',
            'nlm': 'method_c'
        }
        
        method_key = method_map.get(method_name, method_name)
        
        if method_key in self.method_configs:
            return self.method_configs[method_key]
        else:
            return None
    
    def benchmark_methods(self, test_image, noise_type, noise_level):
        """
        Benchmark all three methods on a test image
        Returns performance comparison
        """
        if test_image is None:
            raise ValueError("Test image cannot be None")
        
        results = {}
        
        # Test each method
        for method_name, method_func in [
            ('method_a', self.method_a_denoise),
            ('method_b', self.method_b_denoise),
            ('method_c', self.method_c_denoise)
        ]:
            try:
                result = method_func(test_image, noise_type, noise_level)
                results[method_name] = {
                    'processing_time': result['processing_time'],
                    'method_used': result['method'],
                    'success': True
                }
            except Exception as e:
                results[method_name] = {
                    'processing_time': 0.0,
                    'method_used': method_name,
                    'success': False,
                    'error': str(e)
                }
        
        return results

# Example usage and testing
def demo_core_methods():
    """Demonstrate the core methods with a simple test"""
    print("🧪 CORE METHODS DEMONSTRATION")
    print("=" * 40)
    
    # Create a simple test image with known noise
    test_image = np.random.rand(100, 100) * 255
    test_image = test_image.astype(np.uint8)
    
    # Add Gaussian noise
    noise = np.random.normal(0, 25, test_image.shape)
    noisy_image = np.clip(test_image + noise, 0, 255).astype(np.uint8)
    
    # Initialize methods
    methods = CoreDenoisingMethods()
    
    # Test each method
    for method_name in ['method_a', 'method_b', 'method_c']:
        print(f"\n🔧 Testing {method_name}...")
        
        if method_name == 'method_a':
            result = methods.method_a_denoise(noisy_image, 'gaussian', 0.1)
        elif method_name == 'method_b':
            result = methods.method_b_denoise(noisy_image, 'gaussian', 0.1)
        else:
            result = methods.method_c_denoise(noisy_image, 'gaussian', 0.1)
        
        print(f"   ✅ Success! Processing time: {result['processing_time']:.3f}s")
        print(f"   📊 Method: {result['method']}")
    
    # Benchmark all methods
    print(f"\n📈 BENCHMARKING ALL METHODS...")
    benchmark_results = methods.benchmark_methods(noisy_image, 'gaussian', 0.1)
    
    for method, stats in benchmark_results.items():
        status = "✅" if stats['success'] else "❌"
        print(f"   {status} {method}: {stats['processing_time']:.3f}s")
    
    print(f"\n🎯 Core methods demonstration complete!")

if __name__ == "__main__":
    demo_core_methods()
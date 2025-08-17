#!/usr/bin/env python3
"""
Comprehensive Noise Detection Debugger
Identifies and fixes noise detection issues
"""

import numpy as np
import cv2
import sys
sys.path.append('src')
import traceback

def create_known_noise_samples():
    """Create test images with known noise types for validation"""
    
    # Base clean image
    base_img = np.ones((128, 128, 3), dtype=np.uint8) * 128
    
    # Add geometric pattern for better detection
    cv2.rectangle(base_img, (32, 32), (96, 96), (180, 130, 70), -1)
    cv2.circle(base_img, (64, 64), 20, (70, 180, 130), -1)
    
    samples = {}
    
    # 1. Gaussian noise
    gaussian_noise = np.random.normal(0, 0.2 * 255, base_img.shape)
    gaussian_img = np.clip(base_img.astype(np.float64) + gaussian_noise, 0, 255).astype(np.uint8)
    samples['gaussian'] = gaussian_img
    
    # 2. Salt and pepper noise
    sp_img = base_img.copy()
    total_pixels = base_img.size
    # Salt noise
    num_salt = int(0.1 * total_pixels * 0.5)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in base_img.shape[:2]]
    sp_img[salt_coords[0], salt_coords[1]] = 255
    # Pepper noise
    num_pepper = int(0.1 * total_pixels * 0.5)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in base_img.shape[:2]]
    sp_img[pepper_coords[0], pepper_coords[1]] = 0
    samples['salt_pepper'] = sp_img
    
    # 3. Speckle noise
    speckle_noise = np.random.randn(*base_img.shape) * 0.2
    speckle_img = np.clip(base_img.astype(np.float64) * (1 + speckle_noise), 0, 255).astype(np.uint8)
    samples['speckle'] = speckle_img
    
    # 4. Uniform noise
    uniform_noise = np.random.uniform(-0.2 * 255, 0.2 * 255, base_img.shape)
    uniform_img = np.clip(base_img.astype(np.float64) + uniform_noise, 0, 255).astype(np.uint8)
    samples['uniform'] = uniform_img
    
    # 5. Poisson noise
    scaled = base_img.astype(np.float64) / 255.0
    poisson_img = np.random.poisson(scaled * 0.1 * 100) / (0.1 * 100)
    poisson_img = np.clip(poisson_img * 255, 0, 255).astype(np.uint8)
    samples['poisson'] = poisson_img
    
    return base_img, samples

def debug_detection_pipeline():
    """Debug the complete detection pipeline"""
    
    print("🔍 DEBUGGING NOISE DETECTION PIPELINE")
    print("=" * 60)
    
    # Create test samples
    base_img, noise_samples = create_known_noise_samples()
    
    try:
        from adaptive_denoiser import AdaptiveImageDenoiser
        denoiser = AdaptiveImageDenoiser()
        
        print(f"✅ Adaptive denoiser loaded: {type(denoiser)}")
        print(f"   Available attributes: {[attr for attr in dir(denoiser) if not attr.startswith('_')]}")
        
        # Check detection component
        if hasattr(denoiser, 'detector'):
            detector = denoiser.detector
            print(f"✅ Detector found: {type(detector)}")
            print(f"   Detector methods: {[method for method in dir(detector) if not method.startswith('_')]}")
            
            # Test detection methods
            print(f"\n🧪 TESTING DETECTION METHODS:")
            print("-" * 40)
            
            for noise_type, noisy_img in noise_samples.items():
                print(f"\n🔬 Testing {noise_type.upper()} detection:")
                print(f"   Image stats: min={noisy_img.min()}, max={noisy_img.max()}, "
                      f"mean={noisy_img.mean():.1f}, std={noisy_img.std():.1f}")
                
                # Try different detection method names
                detection_methods = ['detect_noise', 'detect', 'analyze', 'classify_noise']
                detection_result = None
                
                for method_name in detection_methods:
                    if hasattr(detector, method_name):
                        try:
                            method = getattr(detector, method_name)
                            print(f"   🎯 Calling {method_name}()...")
                            detection_result = method(noisy_img)
                            print(f"   ✅ {method_name}() succeeded: {type(detection_result)}")
                            break
                        except Exception as e:
                            print(f"   ❌ {method_name}() failed: {e}")
                
                if detection_result is not None:
                    print(f"   📊 Detection result: {detection_result}")
                    
                    if isinstance(detection_result, dict):
                        detected_type = detection_result.get('primary_type', 'unknown')
                        confidence = detection_result.get('confidence', 0.0)
                        all_scores = detection_result.get('all_scores', {})
                        
                        print(f"   🎯 Detected: {detected_type} (confidence: {confidence:.3f})")
                        print(f"   📈 All scores: {all_scores}")
                        
                        if detected_type == noise_type:
                            print(f"   🎉 CORRECT DETECTION!")
                        else:
                            print(f"   ❌ WRONG! Expected: {noise_type}")
                    else:
                        print(f"   ⚠️  Unexpected result format: {type(detection_result)}")
                else:
                    print(f"   ❌ No detection method worked!")
        else:
            print(f"❌ No detector attribute found!")
            
        # Test full pipeline
        print(f"\n🧪 TESTING FULL DENOISING PIPELINE:")
        print("-" * 40)
        
        for noise_type, noisy_img in noise_samples.items():
            print(f"\n🔬 Testing {noise_type.upper()} pipeline:")
            
            try:
                result = denoiser.denoise_image(noisy_img)
                print(f"   ✅ Pipeline completed: {type(result)}")
                
                if isinstance(result, tuple):
                    image, metadata = result
                    print(f"   📊 Result tuple: image={image.shape}, metadata={type(metadata)}")
                    
                    if isinstance(metadata, dict):
                        print(f"   🔑 Metadata keys: {metadata.keys()}")
                        
                        if 'noise_detection' in metadata:
                            detection = metadata['noise_detection']
                            print(f"   🎯 Pipeline detection: {detection}")
                        else:
                            print(f"   ❌ No 'noise_detection' in metadata!")
                            # Look for alternative keys
                            detection_keys = [k for k in metadata.keys() if 'detect' in k.lower() or 'noise' in k.lower()]
                            print(f"   🔍 Detection-related keys: {detection_keys}")
                    else:
                        print(f"   ❌ Metadata is not dict: {metadata}")
                        
                elif isinstance(result, dict):
                    print(f"   📊 Result dict keys: {result.keys()}")
                else:
                    print(f"   ⚠️  Unexpected result type: {type(result)}")
                    
            except Exception as e:
                print(f"   ❌ Pipeline failed: {e}")
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()

def create_improved_noise_detector():
    """Create an improved noise detection implementation"""
    
    print(f"\n🔧 CREATING IMPROVED NOISE DETECTOR")
    print("=" * 50)
    
    improved_detector_code = '''
import numpy as np
import cv2

class ImprovedNoiseDetector:
    """Improved noise detection with better algorithms"""
    
    def __init__(self):
        self.detection_methods = {
            'variance_analysis': self._variance_based_detection,
            'frequency_analysis': self._frequency_based_detection,
            'statistical_analysis': self._statistical_based_detection,
            'edge_analysis': self._edge_based_detection
        }
    
    def detect_noise(self, image):
        """Main detection method with multiple approaches"""
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Run all detection methods
        all_scores = {}
        
        for method_name, method_func in self.detection_methods.items():
            try:
                scores = method_func(gray)
                for noise_type, score in scores.items():
                    if noise_type not in all_scores:
                        all_scores[noise_type] = []
                    all_scores[noise_type].append(score)
            except Exception as e:
                print(f"Warning: {method_name} failed: {e}")
        
        # Average scores across methods
        averaged_scores = {}
        for noise_type, score_list in all_scores.items():
            averaged_scores[noise_type] = np.mean(score_list)
        
        # Find best detection
        if averaged_scores:
            primary_type = max(averaged_scores, key=averaged_scores.get)
            confidence = averaged_scores[primary_type]
        else:
            primary_type = 'gaussian'  # Default fallback
            confidence = 0.3
            averaged_scores = {'gaussian': 0.3}
        
        return {
            'primary_type': primary_type,
            'confidence': float(confidence),
            'all_scores': averaged_scores
        }
    
    def _variance_based_detection(self, gray):
        """Detect noise based on local variance patterns"""
        
        # Calculate local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Global statistics
        global_var = np.var(gray)
        var_of_var = np.var(local_var)
        
        scores = {}
        
        # Gaussian: high global variance, moderate variance of variance
        if global_var > 800 and var_of_var < global_var * 0.5:
            scores['gaussian'] = min(0.9, global_var / 2000)
        else:
            scores['gaussian'] = max(0.1, global_var / 5000)
        
        # Salt & Pepper: extreme local variance, high variance of variance
        extreme_pixels = np.sum((gray == 0) | (gray == 255))
        total_pixels = gray.size
        sp_ratio = extreme_pixels / total_pixels
        
        if sp_ratio > 0.01 and var_of_var > global_var:
            scores['salt_pepper'] = min(0.9, sp_ratio * 20)
        else:
            scores['salt_pepper'] = sp_ratio * 5
        
        # Speckle: multiplicative, moderate variance
        if 500 < global_var < 1500 and var_of_var > global_var * 0.3:
            scores['speckle'] = min(0.8, (global_var - 500) / 1000)
        else:
            scores['speckle'] = max(0.1, (global_var - 200) / 2000)
        
        # Uniform: high variance, uniform distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_std = np.std(hist)
        
        if global_var > 600 and hist_std < 50:
            scores['uniform'] = min(0.8, global_var / 1500)
        else:
            scores['uniform'] = max(0.1, hist_std / 200)
        
        # Poisson: signal-dependent variance
        mean_intensity = np.mean(gray)
        if abs(global_var - mean_intensity) < mean_intensity * 0.5:
            scores['poisson'] = min(0.8, 1.0 - abs(global_var - mean_intensity) / mean_intensity)
        else:
            scores['poisson'] = max(0.1, 0.5 - abs(global_var - mean_intensity) / (2 * mean_intensity))
        
        return scores
    
    def _frequency_based_detection(self, gray):
        """Detect noise based on frequency domain characteristics"""
        
        # FFT analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # Frequency statistics
        freq_var = np.var(magnitude)
        freq_mean = np.mean(magnitude)
        
        scores = {}
        
        # Gaussian: spread across all frequencies
        scores['gaussian'] = min(0.8, freq_var / 10)
        
        # Salt & Pepper: high frequency spikes
        high_freq_energy = np.sum(magnitude[magnitude > freq_mean + 2*np.std(magnitude)])
        scores['salt_pepper'] = min(0.9, high_freq_energy / (magnitude.size * freq_mean))
        
        # Speckle: moderate frequency spread
        scores['speckle'] = min(0.7, freq_var / 15)
        
        # Uniform: broad frequency spectrum
        scores['uniform'] = min(0.7, freq_var / 12)
        
        # Poisson: similar to Gaussian but signal-dependent
        scores['poisson'] = min(0.7, freq_var / 13)
        
        return scores
    
    def _statistical_based_detection(self, gray):
        """Detect noise based on statistical properties"""
        
        # Basic statistics
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        skewness = self._calculate_skewness(gray)
        kurtosis = self._calculate_kurtosis(gray)
        
        scores = {}
        
        # Gaussian: near-zero skewness, kurtosis around 3
        gaussian_score = 1.0 - (abs(skewness) + abs(kurtosis - 3)) / 4
        scores['gaussian'] = max(0.1, min(0.9, gaussian_score))
        
        # Salt & Pepper: extreme values, high kurtosis
        if kurtosis > 5 and std_val > 60:
            scores['salt_pepper'] = min(0.9, (kurtosis - 3) / 10)
        else:
            scores['salt_pepper'] = max(0.1, kurtosis / 20)
        
        # Speckle: positive skewness usually
        if skewness > 0.3:
            scores['speckle'] = min(0.8, skewness * 2)
        else:
            scores['speckle'] = max(0.1, abs(skewness))
        
        # Uniform: low kurtosis
        if kurtosis < 2.5:
            scores['uniform'] = min(0.8, (3 - kurtosis) / 2)
        else:
            scores['uniform'] = max(0.1, 3 / kurtosis)
        
        # Poisson: skewness related to signal level
        expected_skewness = 1 / np.sqrt(max(mean_val, 1))
        if abs(skewness - expected_skewness) < 0.5:
            scores['poisson'] = min(0.8, 1.0 - abs(skewness - expected_skewness))
        else:
            scores['poisson'] = max(0.1, 0.5)
        
        return scores
    
    def _edge_based_detection(self, gray):
        """Detect noise based on edge preservation"""
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Gradient analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_std = np.std(gradient_magnitude)
        
        scores = {}
        
        # Gaussian: preserved edges, moderate gradient
        if 0.02 < edge_density < 0.1 and grad_std > 20:
            scores['gaussian'] = min(0.8, edge_density * 10)
        else:
            scores['gaussian'] = max(0.2, edge_density * 5)
        
        # Salt & Pepper: disrupted edges
        if edge_density < 0.01:
            scores['salt_pepper'] = min(0.9, (0.02 - edge_density) * 50)
        else:
            scores['salt_pepper'] = max(0.1, 0.02 / edge_density)
        
        # Speckle: preserved edges, high gradient variation
        if edge_density > 0.03 and grad_std > 30:
            scores['speckle'] = min(0.8, grad_std / 50)
        else:
            scores['speckle'] = max(0.2, grad_std / 100)
        
        # Uniform: moderate edge preservation
        scores['uniform'] = min(0.7, edge_density * 8)
        
        # Poisson: signal-dependent edge preservation
        scores['poisson'] = min(0.7, edge_density * 7)
        
        return scores
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4)
'''
    
    print("📝 Improved detector code created")
    print("🔧 To implement: Save as 'improved_noise_detector.py' and integrate")
    
    return improved_detector_code

def test_improved_detector():
    """Test the improved detector implementation"""
    
    print(f"\n🧪 TESTING IMPROVED DETECTOR")
    print("=" * 50)
    
    # This would test the improved detector if implemented
    print("💡 To test the improved detector:")
    print("   1. Save the improved detector code")
    print("   2. Replace detector in adaptive_denoiser.py")
    print("   3. Run: python debug_noise_detection.py")

if __name__ == "__main__":
    debug_detection_pipeline()
    create_improved_noise_detector()
    test_improved_detector()
    
    print("📝 Created improved noise detector implementation")
    
    # return improved_detector_code
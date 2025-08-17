#!/usr/bin/env python3
"""
Debug script to identify why noise detection is failing
"""

import numpy as np
import cv2
import sys
sys.path.append('src')

def test_noise_detection():
    """Test noise detection with known noise types"""
    
    print("🔍 DEBUGGING NOISE DETECTION MODULE")
    print("=" * 50)
    
    # Create test image
    test_img = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)
    print(f"✅ Test image created: {test_img.shape}")
    
    # Add different known noise types
    noise_types = {
        'gaussian': add_gaussian_noise(test_img, 0.2),
        'salt_pepper': add_salt_pepper_noise(test_img, 0.1),
        'speckle': add_speckle_noise(test_img, 0.2),
        'uniform': add_uniform_noise(test_img, 0.2),
        'poisson': add_poisson_noise(test_img, 0.1)
    }
    
    try:
        from adaptive_denoiser import AdaptiveImageDenoiser
        denoiser = AdaptiveImageDenoiser()
        
        print(f"\n✅ Adaptive denoiser imported successfully")
        print(f"   Denoiser type: {type(denoiser)}")
        
        # Check if denoiser has noise detection component
        if hasattr(denoiser, 'noise_detector'):
            print(f"   ✅ Has noise_detector: {type(denoiser.noise_detector)}")
            
            # Test noise detection directly
            print(f"\n🧪 TESTING NOISE DETECTION DIRECTLY:")
            print("-" * 40)
            
            for noise_name, noisy_img in noise_types.items():
                print(f"\n🔬 Testing {noise_name.upper()} noise:")
                print(f"   Image shape: {noisy_img.shape}")
                print(f"   Image dtype: {noisy_img.dtype}")
                print(f"   Value range: {noisy_img.min()}-{noisy_img.max()}")
                
                try:
                    # Call noise detection directly
                    detection_result = denoiser.noise_detector.detect_noise(noisy_img)
                    print(f"   ✅ Detection result: {detection_result}")
                    
                    if isinstance(detection_result, dict):
                        detected_type = detection_result.get('primary_type', 'unknown')
                        confidence = detection_result.get('confidence', 0.0)
                        print(f"   📊 Detected: {detected_type} (confidence: {confidence:.3f})")
                        
                        if detected_type == noise_name:
                            print(f"   🎯 CORRECT DETECTION!")
                        else:
                            print(f"   ❌ WRONG! Expected: {noise_name}")
                    else:
                        print(f"   ❌ Invalid result type: {type(detection_result)}")
                        
                except Exception as e:
                    print(f"   ❌ Detection failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"   ❌ No noise_detector attribute found!")
            print(f"   Available attributes: {[attr for attr in dir(denoiser) if not attr.startswith('_')]}")
        
        # Test full denoising pipeline
        print(f"\n🧪 TESTING FULL DENOISING PIPELINE:")
        print("-" * 40)
        
        for noise_name, noisy_img in noise_types.items():
            print(f"\n🔬 Testing {noise_name.upper()} denoising:")
            
            try:
                result = denoiser.denoise_image(noisy_img)
                print(f"   ✅ Denoising completed")
                print(f"   Result type: {type(result)}")
                
                if isinstance(result, dict):
                    if 'metadata' in result and 'noise_detection' in result['metadata']:
                        detection = result['metadata']['noise_detection']
                        print(f"   📊 Pipeline detection: {detection}")
                    else:
                        print(f"   ❌ No detection metadata found")
                        print(f"   Available keys: {result.keys()}")
                elif isinstance(result, np.ndarray):
                    print(f"   ⚠️  Direct array result (no metadata)")
                else:
                    print(f"   ❌ Unexpected result type")
                    
            except Exception as e:
                print(f"   ❌ Pipeline failed: {e}")
                import traceback
                traceback.print_exc()
                
    except ImportError as e:
        print(f"❌ Failed to import adaptive_denoiser: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def add_gaussian_noise(image, noise_level):
    """Add Gaussian noise"""
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(image, noise_level):
    """Add salt and pepper noise"""
    noisy = image.copy()
    total_pixels = image.size
    
    # Salt noise
    num_salt = int(noise_level * total_pixels * 0.5)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
    noisy[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper noise
    num_pepper = int(noise_level * total_pixels * 0.5)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape[:2]]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy

def add_speckle_noise(image, noise_level):
    """Add speckle noise"""
    noise = np.random.randn(*image.shape) * noise_level
    noisy = image.astype(np.float64) * (1 + noise)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_uniform_noise(image, noise_level):
    """Add uniform noise"""
    noise = np.random.uniform(-noise_level * 255, noise_level * 255, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_poisson_noise(image, noise_level):
    """Add Poisson noise"""
    scaled = image.astype(np.float64) / 255.0
    noisy = np.random.poisson(scaled * noise_level * 100) / (noise_level * 100)
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    test_noise_detection()
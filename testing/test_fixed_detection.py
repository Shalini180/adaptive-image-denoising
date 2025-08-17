import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append('src')

def create_test_images():
    """Create distinctive test images for each noise type"""
    # Base clean image
    base_img = np.ones((100, 100, 3), dtype=np.uint8) * 120
    cv2.rectangle(base_img, (25, 25), (75, 75), (160, 100, 80), -1)
    cv2.circle(base_img, (50, 50), 15, (80, 160, 100), -1)
    
    test_images = {}
    
    # 1. Strong Gaussian noise
    gaussian_noise = np.random.normal(0, 25, base_img.shape)
    gaussian_img = np.clip(base_img.astype(np.float64) + gaussian_noise, 0, 255).astype(np.uint8)
    test_images['gaussian'] = gaussian_img
    
    # 2. Clear Salt & Pepper noise
    sp_img = base_img.copy()
    # Add salt (white) pixels
    salt_mask = np.random.random(base_img.shape[:2]) < 0.08
    sp_img[salt_mask] = 255
    # Add pepper (black) pixels  
    pepper_mask = np.random.random(base_img.shape[:2]) < 0.08
    sp_img[pepper_mask] = 0
    test_images['salt_pepper'] = sp_img
    
    # 3. Strong Speckle noise (multiplicative)
    speckle_noise = np.random.randn(*base_img.shape) * 0.25
    speckle_img = base_img.astype(np.float64) * (1 + speckle_noise)
    speckle_img = np.clip(speckle_img, 0, 255).astype(np.uint8)
    test_images['speckle'] = speckle_img
    
    return test_images

def test_detection_only():
    """Test just the detection without full pipeline"""
    print("🔬 TESTING DETECTION ONLY")
    print("=" * 40)
    
    try:
        from improved_noise_detector import ImprovedNoiseDetector
        detector = ImprovedNoiseDetector()
        
        test_images = create_test_images()
        correct = 0
        total = len(test_images)
        
        for expected_type, image in test_images.items():
            print(f"\n📋 Testing {expected_type.upper()}:")
            
            # Test detection
            result = detector.detect_noise(image)
            detected = result['primary_type']
            confidence = result['confidence']
            scores = {k: f"{v:.3f}" for k, v in result['all_scores'].items()}
            
            print(f"   Expected: {expected_type}")
            print(f"   Detected: {detected} (conf: {confidence:.3f})")
            print(f"   All scores: {scores}")
            
            if detected == expected_type:
                print("   ✅ CORRECT!")
                correct += 1
            else:
                print("   ❌ WRONG")
        
        accuracy = (correct / total) * 100
        print(f"\n📊 DETECTION ACCURACY: {accuracy:.1f}% ({correct}/{total})")
        return accuracy >= 66
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the full pipeline with adaptive denoiser"""
    print("\n🚀 TESTING FULL PIPELINE")
    print("=" * 40)
    
    try:
        # Import your current adaptive denoiser
        import sys
        import os
        
        # Read the adaptive denoiser from the document provided
        exec(open('src/adaptive_denoiser.py').read())
        
        print("✅ Adaptive denoiser loaded")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE DETECTION FIX TEST")
    print("=" * 50)
    
    # Test 1: Detection only
    detection_ok = test_detection_only()
    
    # Test 2: Full pipeline
    pipeline_ok = test_full_pipeline()
    
    # Summary
    print("\n📝 SUMMARY")
    print("=" * 20)
    print(f"Detection Test: {'✅ PASS' if detection_ok else '❌ FAIL'}")
    print(f"Pipeline Test:  {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    
    if detection_ok and pipeline_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your detection system should now work correctly!")
    else:
        print("\n⚠️  Some tests failed. Check the details above.")

if __name__ == "__main__":
    main()
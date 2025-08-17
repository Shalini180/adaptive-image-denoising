import sys
import os
sys.path.append('src')

def check_detector_version():
    """Check which version of the detector is installed"""
    
    print("🔍 CHECKING DETECTOR VERSION")
    print("=" * 35)
    
    try:
        from improved_noise_detector import ImprovedNoiseDetector
        detector = ImprovedNoiseDetector()
        
        # Test 1: Check if new methods exist
        print("📋 Method Availability:")
        
        has_estimate_noise = hasattr(detector, 'estimate_noise_level')
        print(f"   estimate_noise_level(): {'✅ EXISTS' if has_estimate_noise else '❌ MISSING'}")
        
        has_detect_noise_type = hasattr(detector, 'detect_noise_type')
        print(f"   detect_noise_type(): {'✅ EXISTS' if has_detect_noise_type else '❌ MISSING'}")
        
        # Test 2: Check return format
        print(f"\n🔬 Testing Return Format:")
        
        import numpy as np
        import cv2
        test_img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = detector.detect_noise(test_img)
        
        has_both_keys = 'primary_type' in result and 'primary_noise_type' in result
        print(f"   Both primary_type & primary_noise_type: {'✅ YES' if has_both_keys else '❌ NO'}")
        
        confidence = result.get('confidence', 0)
        print(f"   Confidence value: {confidence:.3f}")
        
        # Test 3: Check for bias (confidence should not be exactly 1.0)
        if confidence == 1.0:
            print(f"   🚨 BIAS DETECTED: Confidence is exactly 1.0 (old version)")
            version = "OLD"
        else:
            print(f"   ✅ BALANCED: Confidence is reasonable (new version)")
            version = "NEW"
        
        # Test 4: Check scores
        scores = result.get('all_scores', {})
        max_score = max(scores.values()) if scores else 0
        perfect_scores = sum(1 for v in scores.values() if v == 1.0)
        
        print(f"   Max score: {max_score:.3f}")
        print(f"   Perfect scores (1.0): {perfect_scores}")
        
        if perfect_scores > 1:
            print(f"   🚨 MULTIPLE 1.0 SCORES: Old normalization detected")
            version = "OLD"
        
        # Test 5: Check specific method signatures
        print(f"\n🔧 Method Signatures:")
        
        try:
            # Check if _detect_uniform_fixed method exists (new version indicator)
            uniform_method = getattr(detector, '_detect_uniform_fixed', None)
            if uniform_method:
                print(f"   _detect_uniform_fixed(): ✅ NEW VERSION")
                version = "NEW"
            else:
                print(f"   _detect_uniform_fixed(): ❌ OLD VERSION")
                if version != "OLD":
                    version = "OLD"
        except:
            print(f"   Method check failed")
        
        # Final determination
        print(f"\n📊 FINAL ASSESSMENT:")
        print(f"   Detector Version: {version}")
        
        if version == "OLD":
            print(f"   🚨 YOU'RE USING THE OLD VERSION!")
            print(f"   📝 The file needs to be replaced with the new code.")
            return False
        else:
            print(f"   ✅ YOU'RE USING THE NEW VERSION!")
            return True
            
    except Exception as e:
        print(f"❌ Error checking detector: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_file_replacement_steps():
    """Show steps to replace the file"""
    
    print(f"\n🔧 HOW TO REPLACE THE FILE:")
    print(f"=" * 30)
    print(f"1. Open your file: src/improved_noise_detector.py")
    print(f"2. Select ALL content (Ctrl+A)")
    print(f"3. Delete everything")
    print(f"4. Copy the new code from the artifact above")
    print(f"5. Paste into the file (Ctrl+V)")
    print(f"6. Save the file (Ctrl+S)")
    print(f"7. Run this checker again to verify")

if __name__ == "__main__":
    is_new_version = check_detector_version()
    
    if not is_new_version:
        show_file_replacement_steps()
        print(f"\n💡 After replacing the file, run:")
        print(f"   python check_detector_version.py")
        print(f"   python src/verify_detection_fix.py")
    else:
        print(f"\n🎉 NEW VERSION CONFIRMED!")
        print(f"If you're still getting wrong results, the issue might be elsewhere.")
        print(f"Try running: python src/verify_detection_fix.py")
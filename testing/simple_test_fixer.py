import sys
import numpy as np
import cv2
sys.path.append('src')

def test_current_detector():
    """Test what's actually in your current detector file"""
    
    print("🔍 TESTING CURRENT DETECTOR FILE")
    print("=" * 40)
    
    try:
        from improved_noise_detector import ImprovedNoiseDetector
        detector = ImprovedNoiseDetector()
        
        # Create simple test image
        test_img = np.random.randint(100, 150, (50, 50, 3), dtype=np.uint8)
        
        # Run detection 3 times to check consistency
        print("🔄 Testing consistency (3 runs):")
        
        for i in range(3):
            result = detector.detect_noise(test_img)
            scores = result['all_scores']
            detected = result['primary_type']
            confidence = result['confidence']
            
            print(f"   Run {i+1}: {detected} (conf: {confidence:.3f})")
            print(f"          Scores: {[f'{k}:{v:.2f}' for k,v in scores.items()]}")
        
        # Check method signatures
        print(f"\n🔧 Method Check:")
        has_balanced = hasattr(detector, '_detect_speckle_balanced')
        print(f"   Has _detect_speckle_balanced(): {'✅ YES' if has_balanced else '❌ NO'}")
        
        has_normalize_balanced = hasattr(detector, '_normalize_scores_balanced')
        print(f"   Has _normalize_scores_balanced(): {'✅ YES' if has_normalize_balanced else '❌ NO'}")
        
        if not has_balanced or not has_normalize_balanced:
            print(f"\n🚨 OLD VERSION DETECTED!")
            print(f"   Your file wasn't updated properly.")
            return False
        else:
            print(f"\n✅ NEW VERSION CONFIRMED!")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_manual_update_steps():
    """Show how to manually update the file"""
    
    print(f"\n🔧 MANUAL UPDATE STEPS:")
    print(f"=" * 25)
    print(f"1. Open: src/improved_noise_detector.py")
    print(f"2. Check the file size - it should be about 15-20 KB")
    print(f"3. Search for '_detect_speckle_balanced' - this method should exist")
    print(f"4. If not found, the file wasn't updated properly")
    print(f"5. Copy the ENTIRE new code and replace ALL content")
    print(f"6. Save and test again")

if __name__ == "__main__":
    success = test_current_detector()
    
    if not success:
        show_manual_update_steps()
        
        print(f"\n💡 QUICK FIX OPTION:")
        print(f"If you're having trouble updating the file,")
        print(f"try renaming the old file first:")
        print(f"   mv src/improved_noise_detector.py src/old_detector_backup.py")
        print(f"Then create a new file with the new code.")
    else:
        print(f"\n🎯 FILE IS UPDATED!")
        print(f"The inconsistent results suggest the algorithms")
        print(f"need final tuning. Let me provide a targeted fix...")
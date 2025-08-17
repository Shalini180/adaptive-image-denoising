"""
Simple Working Test
Tests the fixed adaptive denoising system
"""

import numpy as np
import cv2
import sys
import time
sys.path.append('src')

try:
    from adaptive_denoiser import AdaptiveImageDenoiser
    print("✅ Import successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_simple_denoising():
    """Test simple denoising functionality"""
    
    print("🎯 SIMPLE DENOISING TEST")
    print("=" * 30)
    
    try:
        # Create a simple test image
        print("📸 Creating test image...")
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (100, 100), (150, 100, 50), -1)
        cv2.circle(img, (64, 64), 30, (50, 150, 100), -1)
        
        # Add Gaussian noise
        print("🎲 Adding noise...")
        noise = np.random.normal(0, 20, img.shape)
        noisy_img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        
        # Initialize denoiser
        print("🔧 Initializing denoiser...")
        denoiser = AdaptiveImageDenoiser()
        
        # Apply denoising
        print("🎯 Applying denoising...")
        start_time = time.time()
        
        result = denoiser.denoise_image(noisy_img)
        
        elapsed = time.time() - start_time
        
        # Check results
        if result and 'final_image' in result and result['final_image'] is not None:
            print(f"   ✅ Success! ({elapsed:.3f}s)")
            
            # Calculate basic metrics
            mse_noisy = np.mean((img.astype(np.float64) - noisy_img.astype(np.float64))**2)
            mse_denoised = np.mean((img.astype(np.float64) - result['final_image'].astype(np.float64))**2)
            
            psnr_noisy = 20 * np.log10(255.0 / np.sqrt(mse_noisy)) if mse_noisy > 0 else float('inf')
            psnr_denoised = 20 * np.log10(255.0 / np.sqrt(mse_denoised)) if mse_denoised > 0 else float('inf')
            
            print(f"   Input PSNR: {psnr_noisy:.2f} dB")
            print(f"   Output PSNR: {psnr_denoised:.2f} dB")
            print(f"   Improvement: {psnr_denoised - psnr_noisy:+.2f} dB")
            
            if 'metadata' in result:
                metadata = result['metadata']
                if 'noise_detection' in metadata:
                    detection = metadata['noise_detection']
                    print(f"   Detected: {detection['primary_type']} (conf: {detection['confidence']:.3f})")
            
            # Save results
            cv2.imwrite('test_original.png', img)
            cv2.imwrite('test_noisy.png', noisy_img)
            cv2.imwrite('test_denoised.png', result['final_image'])
            print("   💾 Images saved: test_*.png")
            
            return True
            
        else:
            print("   ❌ No output from denoising")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_denoising()
    if success:
        print("\n🎉 System is working correctly!")
    else:
        print("\n💥 System still has issues")

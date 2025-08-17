"""
Working Demo of Adaptive Denoising System
Simplified version that focuses on core functionality
"""

import numpy as np
import cv2
import sys
import os
sys.path.append('src')

from adaptive_denoiser import AdaptiveImageDenoiser
import time

def create_test_image():
    """Create a simple test image"""
    # Create a test image with various patterns
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add some geometric patterns
    cv2.rectangle(img, (50, 50), (150, 150), (200, 100, 50), -1)
    cv2.circle(img, (200, 200), 40, (50, 200, 100), -1)
    cv2.line(img, (0, 0), (255, 255), (100, 50, 200), 3)
    
    # Add some texture
    noise_texture = np.random.randint(0, 50, (256, 256, 3))
    img = cv2.add(img, noise_texture.astype(np.uint8))
    
    return img

def add_gaussian_noise(image, noise_level=0.15):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def calculate_psnr(clean, denoised):
    """Calculate PSNR"""
    mse = np.mean((clean.astype(np.float64) - denoised.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    """Run working demo"""
    
    print("🎯 WORKING ADAPTIVE DENOISING DEMO")
    print("=" * 40)
    
    try:
        # Initialize system
        print("🔧 Initializing adaptive denoising system...")
        denoiser = AdaptiveImageDenoiser()
        print("   ✅ System initialized successfully")
        
        # Create test image
        print("\n📸 Creating test image...")
        clean_image = create_test_image()
        print("   ✅ Test image created")
        
        # Add noise
        print("\n🎲 Adding Gaussian noise...")
        noisy_image = add_gaussian_noise(clean_image, 0.15)
        print("   ✅ Noise added")
        
        # Apply denoising
        print("\n🎯 Applying adaptive denoising...")
        start_time = time.time()
        
        result = denoiser.denoise_image(noisy_image)
        
        processing_time = time.time() - start_time
        print(f"   ✅ Denoising completed in {processing_time:.3f}s")
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        
        if result['final_image'] is not None:
            psnr_noisy = calculate_psnr(clean_image, noisy_image)
            psnr_denoised = calculate_psnr(clean_image, result['final_image'])
            improvement = psnr_denoised - psnr_noisy
            
            print(f"\n🎯 RESULTS:")
            print(f"   Detected Noise: {result['metadata']['noise_detection']['primary_type']}")
            print(f"   Detection Confidence: {result['metadata']['noise_detection']['confidence']:.3f}")
            print(f"   PSNR (Noisy): {psnr_noisy:.2f} dB")
            print(f"   PSNR (Denoised): {psnr_denoised:.2f} dB")
            print(f"   Improvement: {improvement:+.2f} dB")
            print(f"   Processing Time: {processing_time:.3f}s")
            print(f"   Refinement Applied: {'✅ YES' if result['metadata']['refinement_applied'] else '❌ NO'}")
            
            # Show processing stages
            print(f"\n⏱️  PROCESSING STAGES:")
            for stage in result['processing_stages']:
                print(f"   {stage['stage'].replace('_', ' ').title()}: {stage['processing_time']:.3f}s")
            
            if improvement > 0:
                print(f"\n🎉 SUCCESS: Adaptive denoising improved image quality!")
            else:
                print(f"\n⚠️  Note: Improvement may be limited on synthetic test image")
                
            # Save results
            cv2.imwrite('demo_clean.png', clean_image)
            cv2.imwrite('demo_noisy.png', noisy_image)
            cv2.imwrite('demo_denoised.png', result['final_image'])
            print(f"\n💾 Images saved:")
            print(f"   • demo_clean.png")
            print(f"   • demo_noisy.png") 
            print(f"   • demo_denoised.png")
            
        else:
            print("   ❌ Denoising failed - no output image")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

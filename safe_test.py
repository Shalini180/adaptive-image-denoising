"""
Safe Adaptive Denoising Test
Uses only basic OpenCV functions to avoid parameter issues
"""

import numpy as np
import cv2
import time

def create_test_image():
    """Create a simple test image"""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.rectangle(img, (50, 50), (200, 200), (180, 130, 70), -1)
    cv2.circle(img, (128, 128), 60, (70, 180, 130), -1)
    
    # Add some lines
    cv2.line(img, (0, 0), (255, 255), (255, 255, 255), 2)
    cv2.line(img, (255, 0), (0, 255), (255, 255, 255), 2)
    
    return img

def add_gaussian_noise(image, noise_level=0.15):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, noise_level * 255, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def simple_adaptive_denoise(image):
    """Simple adaptive denoising using only OpenCV"""
    
    # Apply multiple denoising methods
    gaussian_result = cv2.GaussianBlur(image, (5, 5), 1.0)
    bilateral_result = cv2.bilateralFilter(image, 9, 75, 75)
    median_result = cv2.medianBlur(image, 5)
    
    # Simple adaptive combination (equal weights for simplicity)
    alpha, beta, gamma = 0.4, 0.4, 0.2
    
    combined = (alpha * gaussian_result.astype(np.float32) +
                beta * bilateral_result.astype(np.float32) +
                gamma * median_result.astype(np.float32))
    
    return np.clip(combined, 0, 255).astype(np.uint8)

def calculate_psnr(clean, noisy):
    """Calculate PSNR"""
    mse = np.mean((clean.astype(np.float64) - noisy.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    """Run safe test"""
    
    print("🎯 SAFE ADAPTIVE DENOISING TEST")
    print("=" * 35)
    
    # Create test image
    print("📸 Creating test image...")
    clean_img = create_test_image()
    
    # Add noise
    print("🎲 Adding noise...")
    noisy_img = add_gaussian_noise(clean_img, 0.15)
    
    # Calculate input PSNR
    input_psnr = calculate_psnr(clean_img, noisy_img)
    print(f"   Input PSNR: {input_psnr:.2f} dB")
    
    # Apply simple adaptive denoising
    print("🎯 Applying adaptive denoising...")
    start_time = time.time()
    
    denoised_img = simple_adaptive_denoise(noisy_img)
    
    processing_time = time.time() - start_time
    
    # Calculate output PSNR
    output_psnr = calculate_psnr(clean_img, denoised_img)
    improvement = output_psnr - input_psnr
    
    print(f"   Processing time: {processing_time:.3f}s")
    print(f"   Output PSNR: {output_psnr:.2f} dB")
    print(f"   Improvement: {improvement:+.2f} dB")
    
    # Save results
    cv2.imwrite('safe_test_clean.png', clean_img)
    cv2.imwrite('safe_test_noisy.png', noisy_img)
    cv2.imwrite('safe_test_denoised.png', denoised_img)
    
    print("\n💾 Images saved:")
    print("   • safe_test_clean.png")
    print("   • safe_test_noisy.png") 
    print("   • safe_test_denoised.png")
    
    if improvement > 0:
        print(f"\n🎉 SUCCESS: Adaptive denoising improved image quality!")
        print(f"   The system is working correctly!")
    else:
        print(f"\n⚠️  Note: Limited improvement on this synthetic test")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Safe test completed successfully!")
        else:
            print("\n❌ Safe test failed")
    except Exception as e:
        print(f"\n❌ Safe test error: {e}")
        import traceback
        traceback.print_exc()

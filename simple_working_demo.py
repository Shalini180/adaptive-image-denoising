"""
Simple Working Demo - Core Functionality Only
Tests just the adaptive denoising without complex dependencies
"""

import numpy as np
import cv2
import sys
import time
sys.path.append('src')

try:
    from adaptive_denoiser import AdaptiveImageDenoiser
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the src/ directory contains all required files")
    sys.exit(1)

def create_test_image(size=256):
    """Create a simple test image"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add some patterns
    cv2.rectangle(img, (50, 50), (size-50, size-50), (180, 130, 70), -1)
    cv2.circle(img, (size//2, size//2), size//4, (70, 180, 130), -1)
    
    # Add texture
    texture = np.random.randint(0, 30, (size, size, 3))
    img = cv2.add(img, texture.astype(np.uint8))
    
    return img

def add_noise(image, noise_type='gaussian', level=0.15):
    """Add noise to image"""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, level * 255, image.shape)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    elif noise_type == 'salt_pepper':
        noisy = image.copy()
        num_salt = int(level * image.size * 0.5)
        num_pepper = int(level * image.size * 0.5)
        
        # Salt noise
        coords = [np.random.randint(0, i-1, num_salt) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 255
        
        # Pepper noise  
        coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape[:2]]
        noisy[coords[0], coords[1], :] = 0
        
        return noisy
    else:
        return image

def calculate_psnr(clean, noisy):
    """Calculate PSNR"""
    mse = np.mean((clean.astype(np.float64) - noisy.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    """Main demo function"""
    
    print("🎯 SIMPLE ADAPTIVE DENOISING DEMO")
    print("=" * 40)
    
    try:
        # Initialize the denoiser
        print("🔧 Initializing adaptive denoiser...")
        denoiser = AdaptiveImageDenoiser()
        print("   ✅ Denoiser initialized successfully")
        
        # Test scenarios
        test_scenarios = [
            {'noise_type': 'gaussian', 'level': 0.15, 'name': 'Gaussian'},
            {'noise_type': 'salt_pepper', 'level': 0.05, 'name': 'Salt & Pepper'}
        ]
        
        for scenario in test_scenarios:
            print(f"\n📊 Testing {scenario['name']} Noise")
            print("-" * 30)
            
            # Create test image
            clean_img = create_test_image()
            
            # Add noise
            noisy_img = add_noise(clean_img, scenario['noise_type'], scenario['level'])
            
            # Calculate input PSNR
            input_psnr = calculate_psnr(clean_img, noisy_img)
            print(f"   Input PSNR: {input_psnr:.2f} dB")
            
            # Apply adaptive denoising
            start_time = time.time()
            
            try:
                result = denoiser.denoise_image(noisy_img)
                processing_time = time.time() - start_time
                
                if result and 'final_image' in result and result['final_image'] is not None:
                    # Calculate output PSNR
                    output_psnr = calculate_psnr(clean_img, result['final_image'])
                    improvement = output_psnr - input_psnr
                    
                    print(f"   ✅ Processing successful ({processing_time:.3f}s)")
                    print(f"   Output PSNR: {output_psnr:.2f} dB")
                    print(f"   Improvement: {improvement:+.2f} dB")
                    
                    # Show detection results
                    if 'metadata' in result and 'noise_detection' in result['metadata']:
                        detected = result['metadata']['noise_detection']['primary_type']
                        confidence = result['metadata']['noise_detection']['confidence']
                        correct = detected == scenario['noise_type']
                        print(f"   Detected: {detected} (confidence: {confidence:.3f}) {'✅' if correct else '❌'}")
                    
                    # Save results
                    cv2.imwrite(f'demo_{scenario["noise_type"]}_clean.png', clean_img)
                    cv2.imwrite(f'demo_{scenario["noise_type"]}_noisy.png', noisy_img)
                    cv2.imwrite(f'demo_{scenario["noise_type"]}_denoised.png', result['final_image'])
                    
                else:
                    print("   ❌ Denoising failed - no output image")
                    
            except Exception as e:
                print(f"   ❌ Denoising error: {e}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Result images saved with 'demo_' prefix")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
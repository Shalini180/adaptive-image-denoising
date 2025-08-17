#!/usr/bin/env python3
"""
Simple debug script to find array comparison error
"""

import numpy as np
import sys
import traceback

# Add src to path
sys.path.append('src')

def add_noise(image, noise_type, noise_level):
    """Add noise exactly like the comprehensive tester does"""
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    elif noise_type == 'salt_pepper':
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
        
    elif noise_type == 'speckle':
        noise = np.random.randn(*image.shape) * noise_level
        noisy = image.astype(np.float64) * (1 + noise)
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level * 255, noise_level * 255, image.shape)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
        
    elif noise_type == 'poisson':
        # Poisson noise simulation
        scaled = image.astype(np.float64) / 255.0
        noisy = np.random.poisson(scaled * noise_level * 100) / (noise_level * 100)
        return np.clip(noisy * 255, 0, 255).astype(np.uint8)
        
    else:
        return image

def create_test_images():
    """Create test images exactly like the comprehensive tester"""
    
    test_images = []
    
    # Test different sizes like the comprehensive tester
    for size in [(128, 128), (256, 256)]:
        h, w = size
        
        # Geometric pattern
        img1 = np.zeros((h, w, 3), dtype=np.uint8)
        import cv2
        cv2.rectangle(img1, (w//4, h//4), (3*w//4, 3*h//4), (180, 130, 70), -1)
        cv2.circle(img1, (w//2, h//2), min(w, h)//6, (70, 180, 130), -1)
        test_images.append(('geometric', img1))
        
        # Texture pattern
        img2 = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(0, h, 16):
            for j in range(0, w, 16):
                if (i//16 + j//16) % 2 == 0:
                    img2[i:i+16, j:j+16] = [150, 150, 150]
                else:
                    img2[i:i+16, j:j+16] = [100, 100, 100]
        
        # Add texture noise
        texture_noise = np.random.randint(0, 30, (h, w, 3))
        img2 = cv2.add(img2, texture_noise.astype(np.uint8))
        test_images.append(('texture', img2))
        
        # Natural-like pattern
        img3 = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                img3[i, j] = [
                    int(128 + 64 * np.sin(2 * np.pi * i / 32) * np.cos(2 * np.pi * j / 32)),
                    int(128 + 64 * np.cos(2 * np.pi * i / 48) * np.sin(2 * np.pi * j / 48)),
                    int(128 + 64 * np.sin(2 * np.pi * (i + j) / 64))
                ]
        img3 = np.clip(img3, 0, 255).astype(np.uint8)
        test_images.append(('natural', img3))
    
    return test_images

def find_array_error():
    """Find the exact location of the array error using comprehensive test scenarios"""
    
    print("🔍 FINDING ARRAY COMPARISON ERROR...")
    print("=" * 50)
    
    # Test exactly like the comprehensive tester
    noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
    noise_levels = [0.10, 0.20]
    
    # Create test images
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images")
    
    from adaptive_denoiser import AdaptiveImageDenoiser
    
    for noise_type in noise_types:
        print(f"\n🧪 Testing {noise_type.upper()} noise...")
        
        for noise_level in noise_levels:
            print(f"   Level {noise_level}: ", end="")
            
            for image_name, clean_image in test_images:
                try:
                    # Add noise exactly like the tester
                    noisy_image = add_noise(clean_image, noise_type, noise_level)
                    
                    # Test the adaptive system
                    denoiser = AdaptiveImageDenoiser()
                    result = denoiser.denoise_image(noisy_image)
                    
                    print("✅", end="")
                    
                except Exception as e:
                    if "truth value" in str(e) or "ambiguous" in str(e):
                        print(f"\n\n🚨 FOUND THE ARRAY ERROR! 🚨")
                        print(f"Noise type: {noise_type}")
                        print(f"Noise level: {noise_level}")
                        print(f"Image: {image_name} {clean_image.shape}")
                        print(f"Error: {e}")
                        print("\n📍 EXACT LOCATION:")
                        print("-" * 60)
                        traceback.print_exc()
                        print("-" * 60)
                        return  # Stop at first error
                    else:
                        print("⚠️", end="")
            
            print()  # New line after each level
    
    print("\n✅ No array error found in comprehensive test")
    print("   The error might be in a different component or state-dependent")

if __name__ == "__main__":
    find_array_error()
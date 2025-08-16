"""
Systematic Noise Generation System
Phase 1.2: Generate comprehensive noisy training pairs

Noise Types: Gaussian, Salt-Pepper, Speckle, Uniform, Poisson
Levels: 6 intensity levels per type (0.05 to 0.30)
Validation: >95% accuracy in noise characteristics
"""

import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class NoiseGenerator:
    """
    Systematic noise generation with validation and quality assurance
    Generates realistic noise patterns matching theoretical distributions
    """
    
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
        self.noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        
        # Noise generation parameters
        self.noise_params = {
            'gaussian': {
                'description': 'Additive white Gaussian noise',
                'sigma_scale': 255.0,  # Scale for 8-bit images
                'validation_test': 'normality'
            },
            'salt_pepper': {
                'description': 'Impulse noise (salt and pepper)',
                'salt_vs_pepper': 0.5,  # Equal probability
                'validation_test': 'impulse_ratio'
            },
            'speckle': {
                'description': 'Multiplicative speckle noise',
                'mean': 1.0,
                'validation_test': 'multiplicative'
            },
            'uniform': {
                'description': 'Additive uniform noise',
                'range_scale': 255.0,
                'validation_test': 'uniformity'
            },
            'poisson': {
                'description': 'Poisson photon noise',
                'scale_factor': 100.0,  # Controls noise intensity
                'validation_test': 'poisson_fit'
            }
        }
        
        # Quality assurance
        self.validation_threshold = 0.95  # 95% accuracy requirement
        self.validation_log = []
        
        print(f"🔧 Noise Generation System Initialized")
        print(f"   Noise Types: {len(self.noise_types)}")
        print(f"   Intensity Levels: {len(self.noise_levels)}")
        print(f"   Validation Threshold: {self.validation_threshold*100}%")
    
    def add_gaussian_noise(self, image, noise_level):
        """Add Gaussian noise with specified standard deviation"""
        if len(image.shape) == 3:
            height, width, channels = image.shape
            noise = np.random.normal(0, noise_level * self.noise_params['gaussian']['sigma_scale'], 
                                   (height, width, channels))
        else:
            height, width = image.shape
            noise = np.random.normal(0, noise_level * self.noise_params['gaussian']['sigma_scale'], 
                                   (height, width))
        
        noisy_image = image.astype(np.float64) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image, noise
    
    def add_salt_pepper_noise(self, image, noise_level):
        """Add salt and pepper impulse noise"""
        noisy_image = image.copy()
        
        # Total pixels to corrupt
        total_pixels = image.shape[0] * image.shape[1]
        num_corrupted = int(total_pixels * noise_level)
        
        # Split between salt and pepper
        num_salt = int(num_corrupted * self.noise_params['salt_pepper']['salt_vs_pepper'])
        num_pepper = num_corrupted - num_salt
        
        # Add salt noise (white pixels)
        if num_salt > 0:
            salt_coords = [
                np.random.randint(0, image.shape[0], num_salt),
                np.random.randint(0, image.shape[1], num_salt)
            ]
            noisy_image[salt_coords[0], salt_coords[1]] = 255
        
        # Add pepper noise (black pixels)
        if num_pepper > 0:
            pepper_coords = [
                np.random.randint(0, image.shape[0], num_pepper),
                np.random.randint(0, image.shape[1], num_pepper)
            ]
            noisy_image[pepper_coords[0], pepper_coords[1]] = 0
        
        # Create noise mask for validation
        noise_mask = np.zeros_like(image)
        if num_salt > 0:
            noise_mask[salt_coords[0], salt_coords[1]] = 255
        if num_pepper > 0:
            noise_mask[pepper_coords[0], pepper_coords[1]] = -255
        
        return noisy_image, noise_mask
    
    def add_speckle_noise(self, image, noise_level):
        """Add multiplicative speckle noise"""
        if len(image.shape) == 3:
            height, width, channels = image.shape
            speckle = np.random.gamma(1.0/noise_level, noise_level, (height, width, channels))
        else:
            height, width = image.shape
            speckle = np.random.gamma(1.0/noise_level, noise_level, (height, width))
        
        # Normalize speckle to have mean 1
        speckle = speckle / np.mean(speckle)
        
        # Apply multiplicative noise
        noisy_image = image.astype(np.float64) * speckle
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image, speckle
    
    def add_uniform_noise(self, image, noise_level):
        """Add additive uniform noise"""
        noise_range = noise_level * self.noise_params['uniform']['range_scale']
        
        if len(image.shape) == 3:
            height, width, channels = image.shape
            noise = np.random.uniform(-noise_range, noise_range, (height, width, channels))
        else:
            height, width = image.shape
            noise = np.random.uniform(-noise_range, noise_range, (height, width))
        
        noisy_image = image.astype(np.float64) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image, noise
    
    def add_poisson_noise(self, image, noise_level):
        """Add Poisson photon noise"""
        # Scale image to increase photon count (reduces relative noise)
        scale_factor = self.noise_params['poisson']['scale_factor'] / noise_level
        
        # Convert to photon counts
        scaled_image = image.astype(np.float64) * scale_factor / 255.0
        
        # Apply Poisson noise
        noisy_scaled = np.random.poisson(scaled_image)
        
        # Convert back to [0, 255] range
        noisy_image = (noisy_scaled * 255.0 / scale_factor).astype(np.uint8)
        
        # Calculate actual noise
        noise = noisy_image.astype(np.float64) - image.astype(np.float64)
        
        return noisy_image, noise
    
    def validate_noise_characteristics(self, original_image, noisy_image, noise_data, 
                                     noise_type, noise_level):
        """Validate that generated noise matches expected characteristics"""
        
        if noise_type == 'gaussian':
            return self._validate_gaussian_noise(noise_data, noise_level)
        elif noise_type == 'salt_pepper':
            return self._validate_salt_pepper_noise(original_image, noisy_image, noise_level)
        elif noise_type == 'speckle':
            return self._validate_speckle_noise(noise_data, noise_level)
        elif noise_type == 'uniform':
            return self._validate_uniform_noise(noise_data, noise_level)
        elif noise_type == 'poisson':
            return self._validate_poisson_noise(original_image, noisy_image, noise_level)
        
        return False
    
    def _validate_gaussian_noise(self, noise, expected_sigma):
        """Validate Gaussian noise characteristics"""
        if len(noise.shape) == 3:
            noise_flat = noise.flatten()
        else:
            noise_flat = noise.flatten()
        
        # Test normality
        _, p_value = stats.normaltest(noise_flat)
        
        # Test standard deviation
        actual_sigma = np.std(noise_flat)
        expected_sigma_scaled = expected_sigma * self.noise_params['gaussian']['sigma_scale']
        sigma_error = abs(actual_sigma - expected_sigma_scaled) / expected_sigma_scaled
        
        # Test mean (should be close to 0)
        actual_mean = np.mean(noise_flat)
        mean_error = abs(actual_mean) / expected_sigma_scaled
        
        # Validation criteria
        normality_valid = p_value > 0.01  # Accept if not significantly non-normal
        sigma_valid = sigma_error < 0.1    # Within 10% of expected
        mean_valid = mean_error < 0.1      # Mean close to zero
        
        return {
            'valid': normality_valid and sigma_valid and mean_valid,
            'normality_p_value': p_value,
            'sigma_error': sigma_error,
            'mean_error': mean_error,
            'details': {
                'expected_sigma': expected_sigma_scaled,
                'actual_sigma': actual_sigma,
                'actual_mean': actual_mean
            }
        }
    
    def _validate_salt_pepper_noise(self, original, noisy, expected_ratio):
        """Validate salt and pepper noise characteristics"""
        # Find corrupted pixels
        diff = noisy.astype(np.int16) - original.astype(np.int16)
        
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)  # Average across channels
        
        salt_pixels = np.sum(diff > 200)  # Nearly white
        pepper_pixels = np.sum(diff < -200)  # Nearly black
        total_corrupted = salt_pixels + pepper_pixels
        total_pixels = diff.shape[0] * diff.shape[1]
        
        actual_ratio = total_corrupted / total_pixels
        ratio_error = abs(actual_ratio - expected_ratio) / expected_ratio
        
        # Check salt vs pepper balance
        if total_corrupted > 0:
            salt_pepper_ratio = salt_pixels / total_corrupted
            balance_error = abs(salt_pepper_ratio - 0.5)
        else:
            balance_error = 0.0
        
        return {
            'valid': ratio_error < 0.1 and balance_error < 0.1,
            'ratio_error': ratio_error,
            'balance_error': balance_error,
            'details': {
                'expected_ratio': expected_ratio,
                'actual_ratio': actual_ratio,
                'salt_pixels': salt_pixels,
                'pepper_pixels': pepper_pixels
            }
        }
    
    def _validate_speckle_noise(self, speckle, expected_level):
        """Validate speckle noise characteristics"""
        if len(speckle.shape) == 3:
            speckle_flat = speckle.flatten()
        else:
            speckle_flat = speckle.flatten()
        
        # Speckle should have mean ≈ 1 and std ≈ noise_level
        actual_mean = np.mean(speckle_flat)
        actual_std = np.std(speckle_flat)
        
        mean_error = abs(actual_mean - 1.0)
        std_error = abs(actual_std - expected_level) / expected_level
        
        return {
            'valid': mean_error < 0.1 and std_error < 0.2,
            'mean_error': mean_error,
            'std_error': std_error,
            'details': {
                'expected_mean': 1.0,
                'actual_mean': actual_mean,
                'expected_std': expected_level,
                'actual_std': actual_std
            }
        }
    
    def _validate_uniform_noise(self, noise, expected_level):
        """Validate uniform noise characteristics"""
        if len(noise.shape) == 3:
            noise_flat = noise.flatten()
        else:
            noise_flat = noise.flatten()
        
        expected_range = expected_level * self.noise_params['uniform']['range_scale']
        
        # Test uniformity using Kolmogorov-Smirnov test
        _, p_value = stats.kstest(noise_flat, 
                                 lambda x: stats.uniform.cdf(x, -expected_range, 2*expected_range))
        
        # Test range
        actual_min = np.min(noise_flat)
        actual_max = np.max(noise_flat)
        actual_range = actual_max - actual_min
        range_error = abs(actual_range - 2*expected_range) / (2*expected_range)
        
        return {
            'valid': p_value > 0.01 and range_error < 0.2,
            'uniformity_p_value': p_value,
            'range_error': range_error,
            'details': {
                'expected_range': 2*expected_range,
                'actual_range': actual_range,
                'actual_min': actual_min,
                'actual_max': actual_max
            }
        }
    
    def _validate_poisson_noise(self, original, noisy, expected_level):
        """Validate Poisson noise characteristics"""
        # For Poisson noise, variance should equal mean in photon counts
        diff = noisy.astype(np.float64) - original.astype(np.float64)
        
        if len(diff.shape) == 3:
            diff_flat = diff.flatten()
        else:
            diff_flat = diff.flatten()
        
        # Estimate if noise follows Poisson-like characteristics
        # (This is a simplified validation)
        actual_mean = np.mean(np.abs(diff_flat))
        actual_var = np.var(diff_flat)
        
        # For Poisson noise, variance ≈ mean × scale_factor
        expected_variance_ratio = actual_mean * 2  # Approximate relationship
        variance_error = abs(actual_var - expected_variance_ratio) / max(expected_variance_ratio, 1.0)
        
        return {
            'valid': variance_error < 0.5,  # More lenient for Poisson
            'variance_error': variance_error,
            'details': {
                'actual_mean': actual_mean,
                'actual_var': actual_var,
                'expected_variance_ratio': expected_variance_ratio
            }
        }
    
    def generate_noisy_pair(self, clean_image, noise_type, noise_level):
        """Generate a single noisy image pair with validation"""
        
        # Apply appropriate noise
        if noise_type == 'gaussian':
            noisy_image, noise_data = self.add_gaussian_noise(clean_image, noise_level)
        elif noise_type == 'salt_pepper':
            noisy_image, noise_data = self.add_salt_pepper_noise(clean_image, noise_level)
        elif noise_type == 'speckle':
            noisy_image, noise_data = self.add_speckle_noise(clean_image, noise_level)
        elif noise_type == 'uniform':
            noisy_image, noise_data = self.add_uniform_noise(clean_image, noise_level)
        elif noise_type == 'poisson':
            noisy_image, noise_data = self.add_poisson_noise(clean_image, noise_level)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Validate noise characteristics
        validation_result = self.validate_noise_characteristics(
            clean_image, noisy_image, noise_data, noise_type, noise_level
        )
        
        return {
            'noisy_image': noisy_image,
            'noise_data': noise_data,
            'validation': validation_result,
            'noise_type': noise_type,
            'noise_level': noise_level
        }
    
    def generate_all_noise_variants(self, clean_image_path, output_base_name):
        """Generate all noise variants for a single clean image"""
        
        # Load clean image
        clean_image = cv2.imread(str(clean_image_path))
        if clean_image is None:
            return False, []
        
        generated_pairs = []
        successful_generations = 0
        
        # Generate all combinations
        for noise_type in self.noise_types:
            for noise_level in self.noise_levels:
                
                try:
                    # Generate noisy pair
                    result = self.generate_noisy_pair(clean_image, noise_type, noise_level)
                    
                    # Save if validation passes
                    if result['validation']['valid']:
                        # Create output filename
                        output_filename = f"{output_base_name}_{noise_type}_{noise_level:.2f}.png"
                        output_path = self.dataset_dir / "noisy_images" / noise_type / output_filename
                        
                        # Save noisy image
                        cv2.imwrite(str(output_path), result['noisy_image'])
                        
                        # Log generation data
                        generation_log = {
                            'clean_image': str(clean_image_path),
                            'noisy_image': str(output_path),
                            'noise_type': noise_type,
                            'noise_level': noise_level,
                            'validation_passed': True,
                            'validation_details': result['validation'],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        generated_pairs.append(generation_log)
                        successful_generations += 1
                    
                    else:
                        # Log failed validation
                        self.validation_log.append({
                            'clean_image': str(clean_image_path),
                            'noise_type': noise_type,
                            'noise_level': noise_level,
                            'validation_passed': False,
                            'validation_details': result['validation'],
                            'timestamp': datetime.now().isoformat()
                        })
                
                except Exception as e:
                    print(f"   ❌ Failed to generate {noise_type} noise level {noise_level}: {e}")
        
        expected_pairs = len(self.noise_types) * len(self.noise_levels)
        success_rate = successful_generations / expected_pairs
        
        return success_rate >= self.validation_threshold, generated_pairs
    
    def process_all_clean_images(self):
        """Process all clean images to generate noisy training pairs"""
        print("\n🎲 STARTING NOISE GENERATION PROCESS")
        print("=" * 50)
        
        # Find all clean images
        clean_images = []
        for category in ['photography', 'medical', 'satellite', 'microscopy', 'synthetic', 'smartphone']:
            category_path = self.dataset_dir / "clean_images" / category
            if category_path.exists():
                clean_images.extend(list(category_path.glob("*.png")))
        
        print(f"   📁 Found {len(clean_images)} clean images")
        
        if len(clean_images) == 0:
            print("   ❌ No clean images found! Run dataset collection first.")
            return
        
        # Process each image
        all_generation_logs = []
        successful_images = 0
        total_pairs_generated = 0
        
        for clean_img_path in tqdm(clean_images, desc="Generating noise variants"):
            # Extract base name
            base_name = clean_img_path.stem
            
            # Generate all noise variants
            success, generation_logs = self.generate_all_noise_variants(clean_img_path, base_name)
            
            if success:
                successful_images += 1
                total_pairs_generated += len(generation_logs)
                all_generation_logs.extend(generation_logs)
        
        # Save generation logs
        if all_generation_logs:
            generation_df = pd.DataFrame(all_generation_logs)
            generation_df.to_csv(self.dataset_dir / "metadata" / "noise_generation_log.csv", index=False)
        
        # Save validation logs
        if self.validation_log:
            validation_df = pd.DataFrame(self.validation_log)
            validation_df.to_csv(self.dataset_dir / "metadata" / "validation_failures.csv", index=False)
        
        # Calculate statistics
        success_rate = successful_images / len(clean_images) if clean_images else 0
        expected_total_pairs = len(clean_images) * len(self.noise_types) * len(self.noise_levels)
        
        print(f"\n📊 NOISE GENERATION SUMMARY:")
        print(f"   Clean Images Processed: {len(clean_images)}")
        print(f"   Successful Images: {successful_images}")
        print(f"   Success Rate: {success_rate*100:.1f}%")
        print(f"   Total Pairs Generated: {total_pairs_generated:,}")
        print(f"   Expected Pairs: {expected_total_pairs:,}")
        print(f"   Generation Efficiency: {total_pairs_generated/expected_total_pairs*100:.1f}%")
        
        if success_rate >= self.validation_threshold:
            print(f"   ✅ Quality threshold met ({self.validation_threshold*100}%)")
        else:
            print(f"   ⚠️  Quality threshold not met (target: {self.validation_threshold*100}%)")
        
        return {
            'total_pairs_generated': total_pairs_generated,
            'success_rate': success_rate,
            'successful_images': successful_images,
            'generation_logs': all_generation_logs
        }

def main():
    """Execute noise generation process"""
    print("🎲 NOISE GENERATION SYSTEM")
    print("=" * 40)
    
    # Initialize generator
    generator = NoiseGenerator()
    
    # Process all clean images
    results = generator.process_all_clean_images()
    
    if results and results['total_pairs_generated'] > 0:
        print(f"\n✅ Noise generation complete!")
        print(f"📁 Noisy images saved to: {generator.dataset_dir}/noisy_images/")
        print(f"📊 Logs saved to: {generator.dataset_dir}/metadata/")
    else:
        print(f"\n❌ Noise generation failed or no images found!")

if __name__ == "__main__":
    main()
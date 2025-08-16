"""
Dataset Collection and Management System
Phase 1.2: Comprehensive dataset creation for adaptive denoising training

Target: 10,000+ clean images → 300,000+ noisy training pairs
Strategy: Multiple sources + systematic noise generation + validation
"""

import os
import numpy as np
import cv2
from PIL import Image
import requests
import zipfile
import json
from tqdm import tqdm
import hashlib
from datetime import datetime
import pandas as pd
from pathlib import Path
import shutil

class DatasetCollector:
    """
    Comprehensive dataset collection and management system
    Handles multiple image sources and systematic noise generation
    """
    
    def __init__(self, base_dir="dataset"):
        self.base_dir = Path(base_dir)
        self.setup_directory_structure()
        
        # Target metrics
        self.target_clean_images = 10000
        self.noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
        self.noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        self.target_total_pairs = len(self.noise_types) * len(self.noise_levels) * self.target_clean_images
        
        # Image requirements
        self.min_resolution = (128, 128)
        self.max_resolution = (2048, 2048)
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Dataset tracking
        self.collection_log = []
        self.noise_generation_log = []
        
        print(f"🎯 Dataset Collection Target:")
        print(f"   Clean Images: {self.target_clean_images:,}")
        print(f"   Noise Types: {len(self.noise_types)}")
        print(f"   Noise Levels: {len(self.noise_levels)}")
        print(f"   Total Training Pairs: {self.target_total_pairs:,}")
    
    def setup_directory_structure(self):
        """Create organized directory structure for dataset"""
        directories = [
            'clean_images/photography',
            'clean_images/medical', 
            'clean_images/satellite',
            'clean_images/microscopy',
            'clean_images/synthetic',
            'clean_images/smartphone',
            'noisy_images/gaussian',
            'noisy_images/salt_pepper', 
            'noisy_images/speckle',
            'noisy_images/uniform',
            'noisy_images/poisson',
            'validation/clean',
            'validation/noisy',
            'metadata',
            'downloads/temp'
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Dataset directory structure created: {self.base_dir}")
    
    def download_kodak_dataset(self):
        """Download and process KODAK PhotoCD dataset (24 images)"""
        print("\n📸 Downloading KODAK PhotoCD Dataset...")
        
        kodak_dir = self.base_dir / "downloads" / "kodak"
        kodak_dir.mkdir(exist_ok=True)
        
        # KODAK dataset URLs (publicly available)
        kodak_base_url = "http://r0k.us/graphics/kodak/kodak/"
        kodak_images = [f"kodim{i:02d}.png" for i in range(1, 25)]
        
        downloaded_count = 0
        for img_name in tqdm(kodak_images, desc="Downloading KODAK"):
            try:
                url = kodak_base_url + img_name
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    img_path = kodak_dir / img_name
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Validate and process image
                    if self.validate_and_process_image(img_path, 'photography'):
                        downloaded_count += 1
                    
            except Exception as e:
                print(f"   ❌ Failed to download {img_name}: {e}")
        
        print(f"   ✅ Successfully downloaded {downloaded_count}/24 KODAK images")
        return downloaded_count
    
    def download_div2k_sample(self, sample_size=100):
        """Download sample from DIV2K dataset"""
        print(f"\n🎨 Downloading DIV2K Sample ({sample_size} images)...")
        
        # Note: This is a placeholder for DIV2K dataset access
        # In practice, you would need to register and download from official source
        print("   ℹ️  DIV2K requires registration at: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
        print("   📝 Manual download required - implement after registration")
        
        # For now, create placeholder function
        div2k_dir = self.base_dir / "downloads" / "div2k" 
        div2k_dir.mkdir(exist_ok=True)
        
        # TODO: Implement actual DIV2K download after registration
        return 0
    
    def generate_synthetic_images(self, count=1000):
        """Generate high-quality synthetic images for training"""
        print(f"\n🔬 Generating {count} synthetic images...")
        
        synthetic_dir = self.base_dir / "clean_images" / "synthetic"
        generated_count = 0
        
        # Generate diverse synthetic patterns
        patterns = ['checkerboard', 'gradient', 'texture', 'geometric', 'natural']
        
        for i in tqdm(range(count), desc="Generating synthetic"):
            try:
                pattern_type = patterns[i % len(patterns)]
                img = self.create_synthetic_image(pattern_type, i)
                
                if img is not None:
                    img_path = synthetic_dir / f"synthetic_{pattern_type}_{i:04d}.png"
                    cv2.imwrite(str(img_path), img)
                    
                    if self.validate_and_process_image(img_path, 'synthetic'):
                        generated_count += 1
                        
            except Exception as e:
                print(f"   ❌ Failed to generate synthetic image {i}: {e}")
        
        print(f"   ✅ Generated {generated_count}/{count} synthetic images")
        return generated_count
    
    def create_synthetic_image(self, pattern_type, seed):
        """Create a single synthetic image with specified pattern"""
        np.random.seed(seed)
        
        # Random size within acceptable range
        width = np.random.randint(256, 1024)
        height = np.random.randint(256, 1024)
        
        if pattern_type == 'checkerboard':
            # Create checkerboard pattern
            checker_size = np.random.randint(8, 32)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            for i in range(0, height, checker_size):
                for j in range(0, width, checker_size):
                    if ((i // checker_size) + (j // checker_size)) % 2 == 0:
                        color = np.random.randint(200, 255, 3)
                        img[i:i+checker_size, j:j+checker_size] = color
        
        elif pattern_type == 'gradient':
            # Create smooth gradients
            x = np.linspace(0, 1, width)
            y = np.linspace(0, 1, height)
            X, Y = np.meshgrid(x, y)
            
            # Multiple gradient directions
            gradient = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
            gradient = ((gradient + 1) / 2 * 255).astype(np.uint8)
            
            img = np.stack([gradient, gradient * 0.8, gradient * 0.6], axis=2).astype(np.uint8)
        
        elif pattern_type == 'texture':
            # Create textured patterns
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Apply Gaussian smoothing for texture
            sigma = np.random.uniform(1.0, 5.0)
            for c in range(3):
                img[:, :, c] = cv2.GaussianBlur(img[:, :, c], (15, 15), sigma)
        
        elif pattern_type == 'geometric':
            # Create geometric shapes
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add random circles and rectangles
            num_shapes = np.random.randint(5, 20)
            for _ in range(num_shapes):
                color = np.random.randint(0, 255, 3).tolist()
                
                if np.random.random() > 0.5:
                    # Circle
                    center = (np.random.randint(0, width), np.random.randint(0, height))
                    radius = np.random.randint(10, min(width, height) // 4)
                    cv2.circle(img, center, radius, color, -1)
                else:
                    # Rectangle
                    pt1 = (np.random.randint(0, width), np.random.randint(0, height))
                    pt2 = (np.random.randint(0, width), np.random.randint(0, height))
                    cv2.rectangle(img, pt1, pt2, color, -1)
        
        elif pattern_type == 'natural':
            # Create natural-looking patterns
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Generate Perlin-like noise for natural textures
            for c in range(3):
                channel = np.random.rand(height, width) * 255
                channel = cv2.GaussianBlur(channel.astype(np.uint8), (15, 15), 3.0)
                img[:, :, c] = channel
        
        return img
    
    def validate_and_process_image(self, img_path, category):
        """Validate image quality and add to clean dataset"""
        try:
            # Load and validate image
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            height, width = img.shape[:2]
            
            # Check resolution requirements
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                return False
            
            if width > self.max_resolution[0] or height > self.max_resolution[1]:
                # Resize to maximum resolution
                scale = min(self.max_resolution[0] / width, self.max_resolution[1] / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            # Calculate image quality metrics
            quality_metrics = self.calculate_image_quality(img)
            
            # Quality thresholds
            if quality_metrics['sharpness'] < 10.0:  # Too blurry
                return False
            
            # Generate unique identifier
            img_hash = self.calculate_image_hash(img)
            
            # Save processed image
            processed_path = self.base_dir / "clean_images" / category / f"{img_hash}.png"
            cv2.imwrite(str(processed_path), img)
            
            # Log collection data
            self.collection_log.append({
                'image_hash': img_hash,
                'original_path': str(img_path),
                'processed_path': str(processed_path),
                'category': category,
                'resolution': f"{img.shape[1]}x{img.shape[0]}",
                'quality_metrics': quality_metrics,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to process {img_path}: {e}")
            return False
    
    def calculate_image_quality(self, img):
        """Calculate quality metrics for image validation"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Contrast (standard deviation)
        contrast = gray.std()
        
        # Brightness (mean)
        brightness = gray.mean()
        
        # Noise estimate (high-frequency content)
        noise_estimate = np.mean(np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))
        
        return {
            'sharpness': float(sharpness),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'noise_estimate': float(noise_estimate)
        }
    
    def calculate_image_hash(self, img):
        """Calculate unique hash for duplicate detection"""
        # Resize to small size for hash
        small = cv2.resize(img, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Calculate hash
        img_bytes = gray.tobytes()
        return hashlib.md5(img_bytes).hexdigest()[:16]
    
    def collect_clean_images(self):
        """Execute complete clean image collection process"""
        print("\n🗂️  STARTING CLEAN IMAGE COLLECTION")
        print("=" * 50)
        
        total_collected = 0
        
        # 1. Download KODAK dataset
        kodak_count = self.download_kodak_dataset()
        total_collected += kodak_count
        
        # 2. Generate synthetic images
        synthetic_count = self.generate_synthetic_images(1000)
        total_collected += synthetic_count
        
        # 3. Download DIV2K sample (placeholder)
        div2k_count = self.download_div2k_sample(100)
        total_collected += div2k_count
        
        # Save collection log
        collection_df = pd.DataFrame(self.collection_log)
        collection_df.to_csv(self.base_dir / "metadata" / "clean_collection_log.csv", index=False)
        
        print(f"\n📊 CLEAN IMAGE COLLECTION SUMMARY:")
        print(f"   KODAK Dataset: {kodak_count}")
        print(f"   Synthetic Images: {synthetic_count}")
        print(f"   DIV2K Sample: {div2k_count}")
        print(f"   ⭐ Total Collected: {total_collected}")
        print(f"   🎯 Target: {self.target_clean_images}")
        print(f"   📈 Progress: {total_collected/self.target_clean_images*100:.1f}%")
        
        return total_collected
    
    def get_collection_status(self):
        """Get current collection status and statistics"""
        clean_dirs = [
            'photography', 'medical', 'satellite', 
            'microscopy', 'synthetic', 'smartphone'
        ]
        
        status = {}
        total_clean = 0
        
        for category in clean_dirs:
            category_path = self.base_dir / "clean_images" / category
            if category_path.exists():
                count = len(list(category_path.glob("*.png")))
                status[category] = count
                total_clean += count
        
        # Check noisy images
        total_noisy = 0
        for noise_type in self.noise_types:
            noise_path = self.base_dir / "noisy_images" / noise_type
            if noise_path.exists():
                total_noisy += len(list(noise_path.glob("*.png")))
        
        return {
            'clean_images': status,
            'total_clean': total_clean,
            'total_noisy': total_noisy,
            'completion_percentage': (total_clean / self.target_clean_images) * 100,
            'target_clean': self.target_clean_images,
            'target_total_pairs': self.target_total_pairs
        }

def main():
    """Execute dataset collection process"""
    print("🎯 DATASET COLLECTION SYSTEM")
    print("=" * 40)
    
    # Initialize collector
    collector = DatasetCollector()
    
    # Execute collection
    total_collected = collector.collect_clean_images()
    
    # Show final status
    status = collector.get_collection_status()
    print(f"\n📈 FINAL STATUS:")
    print(f"   Clean Images: {status['total_clean']:,}")
    print(f"   Completion: {status['completion_percentage']:.1f}%")
    
    if total_collected > 0:
        print(f"\n✅ Dataset collection phase 1 complete!")
        print(f"📁 Images saved to: {collector.base_dir}")
        print(f"📊 Next: Run noise generation process")
    else:
        print(f"\n⚠️  No images collected - check network connection")

if __name__ == "__main__":
    main()
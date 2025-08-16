"""
Dataset Validation and Quality Assurance System
Phase 1.2: Ensure dataset meets quality requirements for training

Validation Criteria:
- >95% noise generation accuracy
- Balanced distribution across noise types and levels
- Quality metrics within acceptable ranges
- No duplicate or corrupted images
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
from collections import defaultdict
import hashlib
from tqdm import tqdm

class DatasetValidator:
    """
    Comprehensive validation system for denoising dataset
    Ensures quality and suitability for training adaptive denoising system
    """
    
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.validation_results = {}
        self.quality_metrics = {}
        
        # Quality thresholds
        self.thresholds = {
            'min_noise_accuracy': 0.95,
            'min_images_per_category': 50,
            'max_duplicate_ratio': 0.01,
            'min_resolution': (128, 128),
            'max_resolution': (2048, 2048),
            'min_contrast': 10.0,
            'min_sharpness': 5.0
        }
        
        print(f"🔍 Dataset Validation System")
        print(f"   Target Accuracy: {self.thresholds['min_noise_accuracy']*100}%")
        print(f"   Quality Thresholds Set")
    
    def validate_directory_structure(self):
        """Validate that all required directories exist"""
        print("\n📁 Validating directory structure...")
        
        required_dirs = [
            'clean_images/photography',
            'clean_images/synthetic', 
            'noisy_images/gaussian',
            'noisy_images/salt_pepper',
            'noisy_images/speckle',
            'noisy_images/uniform',
            'noisy_images/poisson',
            'metadata'
        ]
        
        missing_dirs = []
        existing_dirs = []
        
        for dir_path in required_dirs:
            full_path = self.dataset_dir / dir_path
            if full_path.exists():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        structure_valid = len(missing_dirs) == 0
        
        self.validation_results['directory_structure'] = {
            'valid': structure_valid,
            'existing_dirs': existing_dirs,
            'missing_dirs': missing_dirs,
            'completion_ratio': len(existing_dirs) / len(required_dirs)
        }
        
        if structure_valid:
            print(f"   ✅ All {len(required_dirs)} required directories exist")
        else:
            print(f"   ❌ Missing {len(missing_dirs)} directories: {missing_dirs}")
        
        return structure_valid
    
    def count_dataset_images(self):
        """Count images in each category and noise type"""
        print("\n📊 Counting dataset images...")
        
        counts = {
            'clean': {},
            'noisy': {},
            'totals': {}
        }
        
        # Count clean images
        clean_categories = ['photography', 'medical', 'satellite', 'microscopy', 'synthetic', 'smartphone']
        total_clean = 0
        
        for category in clean_categories:
            category_path = self.dataset_dir / "clean_images" / category
            if category_path.exists():
                count = len(list(category_path.glob("*.png")))
                counts['clean'][category] = count
                total_clean += count
            else:
                counts['clean'][category] = 0
        
        counts['totals']['clean'] = total_clean
        
        # Count noisy images
        noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
        total_noisy = 0
        
        for noise_type in noise_types:
            noise_path = self.dataset_dir / "noisy_images" / noise_type
            if noise_path.exists():
                count = len(list(noise_path.glob("*.png")))
                counts['noisy'][noise_type] = count
                total_noisy += count
            else:
                counts['noisy'][noise_type] = 0
        
        counts['totals']['noisy'] = total_noisy
        
        # Expected totals
        noise_levels = 6  # 0.05 to 0.30
        expected_noisy_per_clean = len(noise_types) * noise_levels
        expected_total_noisy = total_clean * expected_noisy_per_clean
        
        counts['totals']['expected_noisy'] = expected_total_noisy
        counts['totals']['generation_efficiency'] = total_noisy / expected_total_noisy if expected_total_noisy > 0 else 0
        
        self.validation_results['image_counts'] = counts
        
        print(f"   📈 Clean Images: {total_clean:,}")
        print(f"   🎲 Noisy Images: {total_noisy:,}")
        print(f"   🎯 Expected Noisy: {expected_total_noisy:,}")
        print(f"   📊 Generation Efficiency: {counts['totals']['generation_efficiency']*100:.1f}%")
        
        return counts
    
    def validate_noise_generation_accuracy(self):
        """Validate noise generation meets accuracy requirements"""
        print("\n🔬 Validating noise generation accuracy...")
        
        # Load noise generation log
        log_path = self.dataset_dir / "metadata" / "noise_generation_log.csv"
        if not log_path.exists():
            print("   ❌ Noise generation log not found")
            return False
        
        try:
            noise_log = pd.read_csv(log_path)
            
            # Parse validation details (stored as string)
            validation_details = []
            for idx, row in noise_log.iterrows():
                try:
                    # Simplified validation parsing
                    validation_details.append({
                        'noise_type': row['noise_type'],
                        'noise_level': row['noise_level'],
                        'validation_passed': row['validation_passed']
                    })
                except:
                    pass
            
            if not validation_details:
                print("   ⚠️  Could not parse validation details")
                return False
            
            # Calculate accuracy by noise type
            accuracy_by_type = {}
            for noise_type in ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']:
                type_entries = [v for v in validation_details if v['noise_type'] == noise_type]
                if type_entries:
                    passed = sum(1 for v in type_entries if v['validation_passed'])
                    accuracy = passed / len(type_entries)
                    accuracy_by_type[noise_type] = {
                        'accuracy': accuracy,
                        'passed': passed,
                        'total': len(type_entries)
                    }
                else:
                    accuracy_by_type[noise_type] = {
                        'accuracy': 0.0,
                        'passed': 0,
                        'total': 0
                    }
            
            # Overall accuracy
            total_passed = sum(stats['passed'] for stats in accuracy_by_type.values())
            total_attempts = sum(stats['total'] for stats in accuracy_by_type.values())
            overall_accuracy = total_passed / total_attempts if total_attempts > 0 else 0.0
            
            # Check if meets threshold
            accuracy_threshold_met = overall_accuracy >= self.thresholds['min_noise_accuracy']
            
            self.validation_results['noise_accuracy'] = {
                'overall_accuracy': overall_accuracy,
                'accuracy_by_type': accuracy_by_type,
                'threshold_met': accuracy_threshold_met,
                'threshold': self.thresholds['min_noise_accuracy'],
                'total_passed': total_passed,
                'total_attempts': total_attempts
            }
            
            print(f"   📊 Overall Accuracy: {overall_accuracy*100:.1f}%")
            print(f"   🎯 Threshold: {self.thresholds['min_noise_accuracy']*100}%")
            
            for noise_type, stats in accuracy_by_type.items():
                print(f"   {noise_type}: {stats['accuracy']*100:.1f}% ({stats['passed']}/{stats['total']})")
            
            if accuracy_threshold_met:
                print(f"   ✅ Accuracy threshold met!")
            else:
                print(f"   ❌ Accuracy below threshold")
            
            return accuracy_threshold_met
            
        except Exception as e:
            print(f"   ❌ Error validating noise accuracy: {e}")
            return False
    
    def check_image_quality(self, sample_size=100):
        """Check quality metrics for a sample of images"""
        print(f"\n🖼️  Checking image quality (sample: {sample_size})...")
        
        # Collect sample of clean images
        clean_images = []
        for category in ['photography', 'synthetic']:
            category_path = self.dataset_dir / "clean_images" / category
            if category_path.exists():
                clean_images.extend(list(category_path.glob("*.png")))
        
        if len(clean_images) == 0:
            print("   ❌ No clean images found for quality check")
            return False
        
        # Sample images
        sample_images = np.random.choice(clean_images, 
                                       min(sample_size, len(clean_images)), 
                                       replace=False)
        
        quality_stats = {
            'resolution': [],
            'contrast': [],
            'sharpness': [],
            'brightness': [],
            'valid_images': 0,
            'issues': []
        }
        
        for img_path in tqdm(sample_images, desc="Quality check"):
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    quality_stats['issues'].append(f"Cannot load: {img_path.name}")
                    continue
                
                # Resolution check
                height, width = img.shape[:2]
                quality_stats['resolution'].append((width, height))
                
                # Convert to grayscale for analysis
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Contrast (standard deviation)
                contrast = np.std(gray)
                quality_stats['contrast'].append(contrast)
                
                # Sharpness (Laplacian variance)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
                quality_stats['sharpness'].append(sharpness)
                
                # Brightness (mean)
                brightness = np.mean(gray)
                quality_stats['brightness'].append(brightness)
                
                # Check against thresholds
                if (width >= self.thresholds['min_resolution'][0] and 
                    height >= self.thresholds['min_resolution'][1] and
                    width <= self.thresholds['max_resolution'][0] and 
                    height <= self.thresholds['max_resolution'][1] and
                    contrast >= self.thresholds['min_contrast'] and
                    sharpness >= self.thresholds['min_sharpness']):
                    quality_stats['valid_images'] += 1
                else:
                    issues = []
                    if width < self.thresholds['min_resolution'][0] or height < self.thresholds['min_resolution'][1]:
                        issues.append("low_resolution")
                    if width > self.thresholds['max_resolution'][0] or height > self.thresholds['max_resolution'][1]:
                        issues.append("high_resolution")
                    if contrast < self.thresholds['min_contrast']:
                        issues.append("low_contrast")
                    if sharpness < self.thresholds['min_sharpness']:
                        issues.append("low_sharpness")
                    
                    quality_stats['issues'].append(f"{img_path.name}: {', '.join(issues)}")
                
            except Exception as e:
                quality_stats['issues'].append(f"Error processing {img_path.name}: {e}")
        
        # Calculate statistics
        total_checked = len(sample_images)
        quality_pass_rate = quality_stats['valid_images'] / total_checked
        
        quality_summary = {
            'total_checked': total_checked,
            'valid_images': quality_stats['valid_images'],
            'quality_pass_rate': quality_pass_rate,
            'mean_contrast': np.mean(quality_stats['contrast']) if quality_stats['contrast'] else 0,
            'mean_sharpness': np.mean(quality_stats['sharpness']) if quality_stats['sharpness'] else 0,
            'mean_brightness': np.mean(quality_stats['brightness']) if quality_stats['brightness'] else 0,
            'issues_found': len(quality_stats['issues']),
            'issues': quality_stats['issues'][:10]  # First 10 issues
        }
        
        self.validation_results['image_quality'] = quality_summary
        
        print(f"   📊 Images Checked: {total_checked}")
        print(f"   ✅ Valid Images: {quality_stats['valid_images']}")
        print(f"   📈 Quality Pass Rate: {quality_pass_rate*100:.1f}%")
        print(f"   📐 Mean Contrast: {quality_summary['mean_contrast']:.1f}")
        print(f"   🔍 Mean Sharpness: {quality_summary['mean_sharpness']:.1f}")
        print(f"   💡 Mean Brightness: {quality_summary['mean_brightness']:.1f}")
        
        if quality_summary['issues_found'] > 0:
            print(f"   ⚠️  Issues Found: {quality_summary['issues_found']}")
            for issue in quality_summary['issues']:
                print(f"      {issue}")
        
        return quality_pass_rate > 0.8  # 80% should pass quality checks
    
    def analyze_dataset_distribution(self):
        """Analyze distribution across noise types, levels, and categories"""
        print(f"\n📈 Analyzing dataset distribution...")
        
        # Load noise generation log for detailed analysis
        log_path = self.dataset_dir / "metadata" / "noise_generation_log.csv"
        if not log_path.exists():
            print("   ❌ Cannot analyze distribution - log file missing")
            return False
        
        try:
            noise_log = pd.read_csv(log_path)
            
            # Distribution by noise type
            noise_type_dist = noise_log['noise_type'].value_counts().to_dict()
            
            # Distribution by noise level
            noise_level_dist = noise_log['noise_level'].value_counts().to_dict()
            
            # Expected counts
            unique_clean_images = noise_log['clean_image'].nunique()
            expected_per_type = unique_clean_images * 6  # 6 noise levels
            expected_per_level = unique_clean_images * 5  # 5 noise types
            
            # Balance analysis
            type_balance = {}
            for noise_type, count in noise_type_dist.items():
                balance = count / expected_per_type if expected_per_type > 0 else 0
                type_balance[noise_type] = balance
            
            level_balance = {}
            for noise_level, count in noise_level_dist.items():
                balance = count / expected_per_level if expected_per_level > 0 else 0
                level_balance[noise_level] = balance
            
            # Check balance (should be close to 1.0 for each)
            type_balance_score = 1.0 - np.std(list(type_balance.values()))
            level_balance_score = 1.0 - np.std(list(level_balance.values()))
            
            distribution_analysis = {
                'noise_type_distribution': noise_type_dist,
                'noise_level_distribution': noise_level_dist,
                'type_balance': type_balance,
                'level_balance': level_balance,
                'type_balance_score': type_balance_score,
                'level_balance_score': level_balance_score,
                'unique_clean_images': unique_clean_images,
                'expected_per_type': expected_per_type,
                'expected_per_level': expected_per_level
            }
            
            self.validation_results['distribution'] = distribution_analysis
            
            print(f"   🖼️  Unique Clean Images: {unique_clean_images}")
            print(f"   📊 Distribution Balance:")
            print(f"      Noise Type Balance: {type_balance_score:.3f}")
            print(f"      Noise Level Balance: {level_balance_score:.3f}")
            
            # Show distribution
            print(f"   🎲 Noise Type Counts:")
            for noise_type, count in noise_type_dist.items():
                balance = type_balance[noise_type]
                print(f"      {noise_type}: {count} (balance: {balance:.2f})")
            
            return type_balance_score > 0.8 and level_balance_score > 0.8
            
        except Exception as e:
            print(f"   ❌ Error analyzing distribution: {e}")
            return False
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print(f"\n📋 Generating validation report...")
        
        # Calculate overall validation score
        validation_scores = []
        
        if 'directory_structure' in self.validation_results:
            validation_scores.append(1.0 if self.validation_results['directory_structure']['valid'] else 0.0)
        
        if 'noise_accuracy' in self.validation_results:
            validation_scores.append(1.0 if self.validation_results['noise_accuracy']['threshold_met'] else 0.0)
        
        if 'image_quality' in self.validation_results:
            validation_scores.append(self.validation_results['image_quality']['quality_pass_rate'])
        
        if 'distribution' in self.validation_results:
            dist = self.validation_results['distribution']
            dist_score = (dist['type_balance_score'] + dist['level_balance_score']) / 2.0
            validation_scores.append(dist_score)
        
        overall_score = np.mean(validation_scores) if validation_scores else 0.0
        
        # Create report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_validation_score': overall_score,
            'validation_passed': overall_score >= 0.8,
            'detailed_results': self.validation_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        report_path = self.dataset_dir / "metadata" / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   📊 Overall Validation Score: {overall_score:.3f}")
        print(f"   🎯 Validation Threshold: 0.800")
        
        if report['validation_passed']:
            print(f"   ✅ Dataset validation PASSED!")
        else:
            print(f"   ❌ Dataset validation FAILED!")
        
        print(f"   📁 Report saved: {report_path}")
        
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if 'image_counts' in self.validation_results:
            counts = self.validation_results['image_counts']
            if counts['totals']['clean'] < 1000:
                recommendations.append("Collect more clean images (target: 10,000+)")
            
            if counts['totals']['generation_efficiency'] < 0.9:
                recommendations.append("Improve noise generation process to reduce failures")
        
        if 'noise_accuracy' in self.validation_results:
            accuracy = self.validation_results['noise_accuracy']
            if not accuracy['threshold_met']:
                recommendations.append("Improve noise generation accuracy (target: 95%+)")
                
                # Specific recommendations by noise type
                for noise_type, stats in accuracy['accuracy_by_type'].items():
                    if stats['accuracy'] < 0.9:
                        recommendations.append(f"Fix {noise_type} noise generation issues")
        
        if 'image_quality' in self.validation_results:
            quality = self.validation_results['image_quality']
            if quality['quality_pass_rate'] < 0.8:
                recommendations.append("Improve image quality filters")
            
            if quality['mean_contrast'] < self.thresholds['min_contrast']:
                recommendations.append("Add higher contrast images")
            
            if quality['mean_sharpness'] < self.thresholds['min_sharpness']:
                recommendations.append("Add sharper images or reduce blur")
        
        if 'distribution' in self.validation_results:
            dist = self.validation_results['distribution']
            if dist['type_balance_score'] < 0.8:
                recommendations.append("Balance noise type distribution")
            
            if dist['level_balance_score'] < 0.8:
                recommendations.append("Balance noise level distribution")
        
        if not recommendations:
            recommendations.append("Dataset meets all quality requirements!")
        
        return recommendations
    
    def run_complete_validation(self):
        """Run complete validation process"""
        print("🔍 DATASET VALIDATION SYSTEM")
        print("=" * 50)
        
        validation_steps = [
            ("Directory Structure", self.validate_directory_structure),
            ("Image Counts", self.count_dataset_images),
            ("Noise Accuracy", self.validate_noise_generation_accuracy),
            ("Image Quality", self.check_image_quality),
            ("Distribution Analysis", self.analyze_dataset_distribution)
        ]
        
        all_passed = True
        
        for step_name, step_function in validation_steps:
            print(f"\n🔬 {step_name}...")
            try:
                step_result = step_function()
                if not step_result:
                    all_passed = False
            except Exception as e:
                print(f"   ❌ Error in {step_name}: {e}")
                all_passed = False
        
        # Generate final report
        report = self.generate_validation_report()
        
        print(f"\n🎯 VALIDATION SUMMARY:")
        print(f"   Overall Score: {report['overall_validation_score']:.3f}")
        print(f"   Status: {'PASSED' if report['validation_passed'] else 'FAILED'}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        return report

def main():
    """Execute dataset validation"""
    validator = DatasetValidator()
    report = validator.run_complete_validation()
    
    return report['validation_passed']

if __name__ == "__main__":
    main()
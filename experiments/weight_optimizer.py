"""
Empirical Weight Optimization System
Phase 1.3: Systematic optimization of method combination weights

Target: Find optimal α, β, γ weights for each noise type
Constraint: α + β + γ = 1.0
Search Space: ~400-500 combinations per noise type
Evaluation: PSNR, SSIM, edge preservation, texture preservation
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
import time
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import our core methods
from core_methods import CoreDenoisingMethods

# Import evaluation metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import filters, feature

class WeightOptimizer:
    """
    Systematic weight optimization for adaptive denoising system
    Finds optimal α, β, γ combination weights for each noise type
    """
    
    def __init__(self, dataset_dir="../dataset", experiment_dir="weight_optimization"):
        self.dataset_dir = Path(dataset_dir)
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Initialize core methods
        self.core_methods = CoreDenoisingMethods()
        
        # Optimization parameters
        self.noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
        self.noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        
        # Weight search space (constraint: α + β + γ = 1.0)
        self.weight_step = 0.05
        self.min_weight = 0.05
        self.max_weight = 0.8
        
        # Generate valid weight combinations
        self.weight_combinations = self._generate_weight_combinations()
        
        # Evaluation metrics configuration
        self.metrics_config = {
            'psnr_weight': 0.3,
            'ssim_weight': 0.3, 
            'edge_weight': 0.2,
            'texture_weight': 0.2
        }
        
        # Optimization results storage
        self.optimization_results = {}
        
        print(f"🎯 EMPIRICAL WEIGHT OPTIMIZATION SYSTEM")
        print(f"=" * 50)
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"🧪 Experiment Dir: {self.experiment_dir}")
        print(f"🔢 Weight Combinations: {len(self.weight_combinations)}")
        print(f"🎲 Noise Types: {len(self.noise_types)}")
        print(f"📊 Total Evaluations: {len(self.weight_combinations) * len(self.noise_types):,}")
    
    def _generate_weight_combinations(self):
        """Generate all valid weight combinations satisfying constraint α + β + γ = 1.0"""
        combinations = []
        
        alpha_range = np.arange(self.min_weight, self.max_weight + self.weight_step, self.weight_step)
        beta_range = np.arange(self.min_weight, self.max_weight + self.weight_step, self.weight_step)
        
        for alpha in alpha_range:
            for beta in beta_range:
                gamma = 1.0 - alpha - beta
                
                # Check if gamma is within valid range
                if self.min_weight <= gamma <= self.max_weight:
                    # Round to avoid floating point precision issues
                    alpha_rounded = round(alpha, 2)
                    beta_rounded = round(beta, 2)
                    gamma_rounded = round(gamma, 2)
                    
                    # Verify constraint (allow small floating point error)
                    if abs(alpha_rounded + beta_rounded + gamma_rounded - 1.0) < 1e-10:
                        combinations.append((alpha_rounded, beta_rounded, gamma_rounded))
        
        print(f"   Generated {len(combinations)} valid weight combinations")
        return combinations
    
    def load_test_images(self, noise_type, sample_size=50):
        """Load clean and noisy image pairs for testing"""
        print(f"\n📁 Loading test images for {noise_type} noise...")
        
        # Find clean images
        clean_images = []
        for category in ['photography', 'synthetic']:
            category_path = self.dataset_dir / "clean_images" / category
            if category_path.exists():
                clean_images.extend(list(category_path.glob("*.png")))
        
        if len(clean_images) == 0:
            raise ValueError("No clean images found in dataset")
        
        # Sample clean images
        if len(clean_images) > sample_size:
            selected_clean = np.random.choice(clean_images, sample_size, replace=False)
        else:
            selected_clean = clean_images
        
        # Load corresponding noisy images
        image_pairs = []
        noisy_dir = self.dataset_dir / "noisy_images" / noise_type
        
        if not noisy_dir.exists():
            raise ValueError(f"Noisy images directory not found: {noisy_dir}")
        
        for clean_path in selected_clean:
            # Find matching noisy images (different noise levels)
            base_name = clean_path.stem
            
            for noise_level in self.noise_levels:
                noisy_name = f"{base_name}_{noise_type}_{noise_level:.2f}.png"
                noisy_path = noisy_dir / noisy_name
                
                if noisy_path.exists():
                    try:
                        # Load images
                        clean_img = cv2.imread(str(clean_path))
                        noisy_img = cv2.imread(str(noisy_path))
                        
                        if clean_img is not None and noisy_img is not None:
                            # Ensure same size
                            if clean_img.shape == noisy_img.shape:
                                image_pairs.append({
                                    'clean': clean_img,
                                    'noisy': noisy_img,
                                    'noise_level': noise_level,
                                    'clean_path': str(clean_path),
                                    'noisy_path': str(noisy_path)
                                })
                    except Exception as e:
                        print(f"   ⚠️  Error loading {noisy_path}: {e}")
        
        print(f"   📊 Loaded {len(image_pairs)} image pairs")
        
        if len(image_pairs) == 0:
            raise ValueError(f"No valid image pairs found for {noise_type}")
        
        return image_pairs
    
    def apply_weighted_combination(self, image, noise_type, noise_level, alpha, beta, gamma):
        """Apply weighted combination of three methods"""
        
        # Apply each method
        result_a = self.core_methods.method_a_denoise(image, noise_type, noise_level)
        result_b = self.core_methods.method_b_denoise(image, noise_type, noise_level)
        result_c = self.core_methods.method_c_denoise(image, noise_type, noise_level)
        
        # Extract denoised images
        denoised_a = result_a['denoised_image'].astype(np.float64)
        denoised_b = result_b['denoised_image'].astype(np.float64)
        denoised_c = result_c['denoised_image'].astype(np.float64)
        
        # Weighted combination
        combined = alpha * denoised_a + beta * denoised_b + gamma * denoised_c
        combined = np.clip(combined, 0, 255).astype(np.uint8)
        
        return {
            'combined': combined,
            'method_a_result': result_a,
            'method_b_result': result_b, 
            'method_c_result': result_c,
            'weights': (alpha, beta, gamma),
            'processing_times': [
                result_a['processing_time'],
                result_b['processing_time'],
                result_c['processing_time']
            ]
        }
    
    def calculate_edge_preservation(self, clean_image, denoised_image):
        """Calculate edge preservation metric"""
        
        # Convert to grayscale if needed
        if len(clean_image.shape) == 3:
            clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
            denoised_gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
        else:
            clean_gray = clean_image
            denoised_gray = denoised_image
        
        # Detect edges using Canny
        clean_edges = cv2.Canny(clean_gray, 50, 150)
        denoised_edges = cv2.Canny(denoised_gray, 50, 150)
        
        # Calculate edge preservation as intersection over union
        intersection = np.logical_and(clean_edges, denoised_edges)
        union = np.logical_or(clean_edges, denoised_edges)
        
        if np.sum(union) == 0:
            return 1.0  # No edges in either image
        
        edge_preservation = np.sum(intersection) / np.sum(union)
        
        return edge_preservation
    
    def calculate_texture_preservation(self, clean_image, denoised_image):
        """Calculate texture preservation metric using local binary patterns"""
        
        try:
            # Convert to grayscale
            if len(clean_image.shape) == 3:
                clean_gray = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
                denoised_gray = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2GRAY)
            else:
                clean_gray = clean_image
                denoised_gray = denoised_image
            
            # Calculate local binary patterns
            radius = 3
            n_points = 8 * radius
            
            clean_lbp = feature.local_binary_pattern(clean_gray, n_points, radius, method='uniform')
            denoised_lbp = feature.local_binary_pattern(denoised_gray, n_points, radius, method='uniform')
            
            # Calculate correlation between LBP histograms
            clean_hist = np.histogram(clean_lbp, bins=n_points + 2, range=(0, n_points + 2))[0]
            denoised_hist = np.histogram(denoised_lbp, bins=n_points + 2, range=(0, n_points + 2))[0]
            
            # Normalize histograms
            clean_hist = clean_hist / np.sum(clean_hist) if np.sum(clean_hist) > 0 else clean_hist
            denoised_hist = denoised_hist / np.sum(denoised_hist) if np.sum(denoised_hist) > 0 else denoised_hist
            
            # Calculate correlation
            correlation = np.corrcoef(clean_hist, denoised_hist)[0, 1]
            
            # Handle NaN case
            if np.isnan(correlation):
                correlation = 0.0
            
            return max(0.0, correlation)  # Ensure non-negative
            
        except Exception as e:
            print(f"   ⚠️  Texture calculation error: {e}")
            return 0.5  # Default moderate score
    
    def evaluate_denoising_performance(self, clean_image, noisy_image, denoised_image):
        """Comprehensive evaluation of denoising performance"""
        
        # Ensure images are in correct format
        if clean_image.dtype != np.uint8:
            clean_image = np.clip(clean_image, 0, 255).astype(np.uint8)
        if denoised_image.dtype != np.uint8:
            denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
        
        # Calculate quantitative metrics
        try:
            # PSNR
            psnr = peak_signal_noise_ratio(clean_image, denoised_image)
            
            # SSIM
            if len(clean_image.shape) == 3:
                ssim = structural_similarity(clean_image, denoised_image, multichannel=True, channel_axis=2, data_range=255)
            else:
                ssim = structural_similarity(clean_image, denoised_image, data_range=255)
            
            # Edge preservation
            edge_preservation = self.calculate_edge_preservation(clean_image, denoised_image)
            
            # Texture preservation  
            texture_preservation = self.calculate_texture_preservation(clean_image, denoised_image)
            
            # Combined score
            combined_score = (
                self.metrics_config['psnr_weight'] * (psnr / 40.0) +  # Normalize PSNR to ~[0,1]
                self.metrics_config['ssim_weight'] * ssim +
                self.metrics_config['edge_weight'] * edge_preservation +
                self.metrics_config['texture_weight'] * texture_preservation
            )
            
            return {
                'psnr': float(psnr),
                'ssim': float(ssim),
                'edge_preservation': float(edge_preservation),
                'texture_preservation': float(texture_preservation),
                'combined_score': float(combined_score)
            }
            
        except Exception as e:
            print(f"   ❌ Error in performance evaluation: {e}")
            return {
                'psnr': 0.0,
                'ssim': 0.0,
                'edge_preservation': 0.0,
                'texture_preservation': 0.0,
                'combined_score': 0.0
            }
    
    def optimize_weights_for_noise_type(self, noise_type, max_images=30):
        """Find optimal weights for a specific noise type"""
        print(f"\n🎯 OPTIMIZING WEIGHTS FOR {noise_type.upper()} NOISE")
        print(f"=" * 50)
        
        # Load test images
        image_pairs = self.load_test_images(noise_type, sample_size=max_images)
        
        # Split into optimization and validation sets
        opt_pairs, val_pairs = train_test_split(image_pairs, test_size=0.3, random_state=42)
        
        print(f"   📊 Optimization set: {len(opt_pairs)} pairs")
        print(f"   📊 Validation set: {len(val_pairs)} pairs")
        
        # Evaluate all weight combinations
        results = []
        
        print(f"   🔄 Testing {len(self.weight_combinations)} weight combinations...")
        
        for alpha, beta, gamma in tqdm(self.weight_combinations, desc=f"Optimizing {noise_type}"):
            
            combination_scores = []
            
            # Test on optimization set
            for pair in opt_pairs:
                try:
                    # Apply weighted combination
                    result = self.apply_weighted_combination(
                        pair['noisy'], noise_type, pair['noise_level'], alpha, beta, gamma
                    )
                    
                    # Evaluate performance
                    metrics = self.evaluate_denoising_performance(
                        pair['clean'], pair['noisy'], result['combined']
                    )
                    
                    combination_scores.append(metrics['combined_score'])
                    
                except Exception as e:
                    print(f"   ⚠️  Error evaluating combination ({alpha}, {beta}, {gamma}): {e}")
                    combination_scores.append(0.0)
            
            # Calculate average score for this combination
            avg_score = np.mean(combination_scores) if combination_scores else 0.0
            
            results.append({
                'alpha': alpha,
                'beta': beta, 
                'gamma': gamma,
                'avg_score': avg_score,
                'std_score': np.std(combination_scores) if combination_scores else 0.0,
                'num_evaluations': len(combination_scores)
            })
        
        # Find best combination
        results.sort(key=lambda x: x['avg_score'], reverse=True)
        best_result = results[0]
        
        print(f"\n   🏆 BEST COMBINATION FOR {noise_type}:")
        print(f"   α = {best_result['alpha']:.2f}")
        print(f"   β = {best_result['beta']:.2f}")  
        print(f"   γ = {best_result['gamma']:.2f}")
        print(f"   Score = {best_result['avg_score']:.4f} ± {best_result['std_score']:.4f}")
        
        # Validate on validation set
        validation_metrics = self._validate_optimal_weights(
            noise_type, val_pairs, 
            best_result['alpha'], best_result['beta'], best_result['gamma']
        )
        
        # Store complete results
        optimization_result = {
            'noise_type': noise_type,
            'optimal_weights': {
                'alpha': best_result['alpha'],
                'beta': best_result['beta'],
                'gamma': best_result['gamma']
            },
            'optimization_score': best_result['avg_score'],
            'optimization_std': best_result['std_score'],
            'validation_metrics': validation_metrics,
            'all_combinations': results[:10],  # Store top 10
            'optimization_set_size': len(opt_pairs),
            'validation_set_size': len(val_pairs),
            'timestamp': datetime.now().isoformat()
        }
        
        self.optimization_results[noise_type] = optimization_result
        
        return optimization_result
    
    def _validate_optimal_weights(self, noise_type, val_pairs, alpha, beta, gamma):
        """Validate optimal weights on independent validation set"""
        print(f"   🔍 Validating on {len(val_pairs)} validation pairs...")
        
        validation_scores = []
        detailed_metrics = {
            'psnr': [],
            'ssim': [], 
            'edge_preservation': [],
            'texture_preservation': [],
            'combined_score': []
        }
        
        for pair in val_pairs:
            try:
                # Apply optimal weights
                result = self.apply_weighted_combination(
                    pair['noisy'], noise_type, pair['noise_level'], alpha, beta, gamma
                )
                
                # Evaluate performance
                metrics = self.evaluate_denoising_performance(
                    pair['clean'], pair['noisy'], result['combined']
                )
                
                # Store detailed metrics
                for metric_name, value in metrics.items():
                    detailed_metrics[metric_name].append(value)
                
            except Exception as e:
                print(f"   ⚠️  Validation error: {e}")
                # Add default scores for failed evaluations
                for metric_name in detailed_metrics:
                    detailed_metrics[metric_name].append(0.0)
        
        # Calculate validation statistics
        validation_summary = {}
        for metric_name, values in detailed_metrics.items():
            validation_summary[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        print(f"   📊 Validation Results:")
        print(f"      PSNR: {validation_summary['psnr']['mean']:.2f} ± {validation_summary['psnr']['std']:.2f}")
        print(f"      SSIM: {validation_summary['ssim']['mean']:.3f} ± {validation_summary['ssim']['std']:.3f}")
        print(f"      Edge: {validation_summary['edge_preservation']['mean']:.3f} ± {validation_summary['edge_preservation']['std']:.3f}")
        print(f"      Texture: {validation_summary['texture_preservation']['mean']:.3f} ± {validation_summary['texture_preservation']['std']:.3f}")
        print(f"      Combined: {validation_summary['combined_score']['mean']:.4f} ± {validation_summary['combined_score']['std']:.4f}")
        
        return validation_summary
    
    def run_complete_optimization(self):
        """Run optimization for all noise types"""
        print(f"🚀 STARTING COMPLETE WEIGHT OPTIMIZATION")
        print(f"=" * 60)
        
        start_time = time.time()
        
        # Optimize each noise type
        for noise_type in self.noise_types:
            try:
                result = self.optimize_weights_for_noise_type(noise_type)
                print(f"   ✅ {noise_type} optimization completed")
                
                # Save intermediate results
                self._save_intermediate_results()
                
            except Exception as e:
                print(f"   ❌ {noise_type} optimization failed: {e}")
                self.optimization_results[noise_type] = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        # Generate final report
        total_time = time.time() - start_time
        final_report = self._generate_optimization_report(total_time)
        
        return final_report
    
    def _save_intermediate_results(self):
        """Save intermediate optimization results"""
        results_path = self.experiment_dir / "optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.optimization_results, f, indent=2, default=str)
    
    def _generate_optimization_report(self, total_time):
        """Generate comprehensive optimization report"""
        
        report = {
            'optimization_summary': {
                'total_processing_time': total_time,
                'noise_types_processed': len(self.optimization_results),
                'successful_optimizations': len([r for r in self.optimization_results.values() if 'optimal_weights' in r]),
                'total_weight_combinations_tested': len(self.weight_combinations),
                'timestamp': datetime.now().isoformat()
            },
            'optimal_weights_summary': {},
            'performance_comparison': {},
            'detailed_results': self.optimization_results
        }
        
        # Extract optimal weights
        for noise_type, result in self.optimization_results.items():
            if 'optimal_weights' in result:
                report['optimal_weights_summary'][noise_type] = result['optimal_weights']
                
                # Performance metrics
                if 'validation_metrics' in result:
                    val_metrics = result['validation_metrics']
                    report['performance_comparison'][noise_type] = {
                        'psnr': val_metrics['psnr']['mean'],
                        'ssim': val_metrics['ssim']['mean'],
                        'combined_score': val_metrics['combined_score']['mean']
                    }
        
        # Save complete report
        report_path = self.experiment_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print(f"=" * 50)
        print(f"⏱️  Total Time: {total_time/3600.0:.2f} hours")
        print(f"📊 Noise Types Processed: {len(self.optimization_results)}")
        
        print(f"\n🏆 OPTIMAL WEIGHTS FOUND:")
        for noise_type, weights in report['optimal_weights_summary'].items():
            print(f"   {noise_type}: α={weights['alpha']:.2f}, β={weights['beta']:.2f}, γ={weights['gamma']:.2f}")
        
        print(f"\n📁 Results saved to: {report_path}")
        
        return report

def main():
    """Execute weight optimization process"""
    print("🎯 EMPIRICAL WEIGHT OPTIMIZATION")
    print("=" * 40)
    
    # Check if dataset exists
    dataset_dir = Path("../dataset")
    if not dataset_dir.exists():
        print("❌ Dataset directory not found. Run Phase 1.2 first.")
        return
    
    # Initialize optimizer
    optimizer = WeightOptimizer(dataset_dir)
    
    # Run complete optimization
    report = optimizer.run_complete_optimization()
    
    # Check success
    successful = report['optimization_summary']['successful_optimizations']
    total = report['optimization_summary']['noise_types_processed']
    
    if successful == total and successful > 0:
        print(f"\n✅ All optimizations completed successfully!")
    else:
        print(f"\n⚠️  Partial success: {successful}/{total} optimizations completed")

if __name__ == "__main__":
    main()
"""
Refinement Strategy System
Phase 1.5: Iterative refinement of uncertain regions

Objectives:
- Identify uncertain regions using optimized uncertainty maps
- Apply more aggressive denoising to uncertain areas
- Optimize refinement parameters for maximum improvement
- Implement adaptive refinement thresholds
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
from scipy import ndimage
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import our systems
from core_methods import CoreDenoisingMethods
from uncertainty_quantifier import UncertaintyQuantifier

class RefinementStrategy:
    """
    Adaptive refinement strategy for uncertain regions
    Implements targeted refinement based on uncertainty predictions
    """
    
    def __init__(self, dataset_dir="../dataset", experiment_dir="refinement_strategy"):
        self.dataset_dir = Path(dataset_dir)
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.core_methods = CoreDenoisingMethods()
        self.uncertainty_quantifier = UncertaintyQuantifier(dataset_dir, "uncertainty_quantification")
        
        # Load optimization results
        self.optimal_weights = self._load_optimal_weights()
        self.uncertainty_weights = self._load_uncertainty_weights()
        
        # Refinement parameters
        self.refinement_config = {
            'uncertainty_thresholds': [70, 75, 80, 85, 90, 95],  # Percentiles
            'refinement_intensity_multipliers': [1.2, 1.4, 1.6, 1.8, 2.0, 2.5],
            'max_refinement_iterations': 3,
            'convergence_threshold': 0.01,
            'blend_kernel_size': 5
        }
        
        # Results storage
        self.refinement_results = {}
        
        print(f"🔧 REFINEMENT STRATEGY SYSTEM")
        print(f"=" * 50)
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"🧪 Experiment Dir: {self.experiment_dir}")
        print(f"📊 Uncertainty Thresholds: {self.refinement_config['uncertainty_thresholds']}")
        print(f"🔄 Max Iterations: {self.refinement_config['max_refinement_iterations']}")
        
        if self.optimal_weights and self.uncertainty_weights:
            print(f"✅ Loaded optimization results for refinement")
        else:
            print(f"⚠️  Some optimization results missing - using defaults")
    
    def _load_optimal_weights(self):
        """Load optimal denoising weights"""
        try:
            with open("weight_optimization/optimization_results.json", 'r') as f:
                results = json.load(f)
            
            optimal_weights = {}
            for noise_type, data in results.items():
                if 'optimal_weights' in data:
                    optimal_weights[noise_type] = data['optimal_weights']
            
            return optimal_weights
        except:
            return None
    
    def _load_uncertainty_weights(self):
        """Load optimal uncertainty weights"""
        try:
            with open("uncertainty_quantification/uncertainty_results.json", 'r') as f:
                results = json.load(f)
            
            uncertainty_weights = {}
            for noise_type, data in results.items():
                if 'optimal_weights' in data:
                    uncertainty_weights[noise_type] = data['optimal_weights']
            
            return uncertainty_weights
        except:
            return None
    
    def identify_uncertain_regions(self, image, noise_type, noise_level, threshold_percentile=80):
        """Identify regions requiring refinement based on uncertainty"""
        
        # Get uncertainty weights for this noise type
        if self.uncertainty_weights and noise_type in self.uncertainty_weights:
            unc_weights = self.uncertainty_weights[noise_type]
            weights = [
                unc_weights['local_variance'],
                unc_weights['edge_proximity'], 
                unc_weights['method_disagreement'],
                unc_weights['snr_uncertainty']
            ]
        else:
            # Default weights
            weights = [0.25, 0.25, 0.25, 0.25]
        
        # Compute uncertainty map
        uncertainty_result = self.uncertainty_quantifier.compute_uncertainty_map(
            image, noise_type, noise_level, weights
        )
        uncertainty_map = uncertainty_result['combined_uncertainty']
        
        # Threshold to identify uncertain regions
        threshold_value = np.percentile(uncertainty_map, threshold_percentile)
        uncertain_mask = uncertainty_map > threshold_value
        
        return {
            'uncertainty_map': uncertainty_map,
            'uncertain_mask': uncertain_mask,
            'threshold_value': threshold_value,
            'uncertain_pixel_ratio': np.sum(uncertain_mask) / uncertain_mask.size,
            'individual_indicators': uncertainty_result['individual_indicators']
        }
    
    def apply_targeted_refinement(self, image, uncertain_mask, noise_type, noise_level, 
                                intensity_multiplier=1.5):
        """Apply more aggressive denoising to uncertain regions"""
        
        # Get base denoising weights
        if self.optimal_weights and noise_type in self.optimal_weights:
            base_weights = self.optimal_weights[noise_type]
            alpha_base = base_weights['alpha']
            beta_base = base_weights['beta']
            gamma_base = base_weights['gamma']
        else:
            # Default weights
            alpha_base, beta_base, gamma_base = 0.33, 0.33, 0.34
        
        # Create refinement weights (more aggressive)
        alpha_refine = min(alpha_base * intensity_multiplier, 0.8)
        beta_refine = min(beta_base * intensity_multiplier, 0.8)
        gamma_refine = min(gamma_base * intensity_multiplier, 0.8)
        
        # Normalize refinement weights
        total_refine = alpha_refine + beta_refine + gamma_refine
        alpha_refine /= total_refine
        beta_refine /= total_refine
        gamma_refine /= total_refine
        
        # Apply base denoising to entire image
        result_a_base = self.core_methods.method_a_denoise(image, noise_type, noise_level)
        result_b_base = self.core_methods.method_b_denoise(image, noise_type, noise_level)
        result_c_base = self.core_methods.method_c_denoise(image, noise_type, noise_level)
        
        base_denoised = (
            alpha_base * result_a_base['denoised_image'].astype(np.float32) +
            beta_base * result_b_base['denoised_image'].astype(np.float32) +
            gamma_base * result_c_base['denoised_image'].astype(np.float32)
        )
        
        # Apply aggressive denoising for refinement
        # Simulate more aggressive parameters by applying methods with modified noise level
        refined_noise_level = noise_level * 1.2  # Assume higher noise for more aggressive treatment
        
        result_a_refine = self.core_methods.method_a_denoise(image, noise_type, refined_noise_level)
        result_b_refine = self.core_methods.method_b_denoise(image, noise_type, refined_noise_level)
        result_c_refine = self.core_methods.method_c_denoise(image, noise_type, refined_noise_level)
        
        refined_denoised = (
            alpha_refine * result_a_refine['denoised_image'].astype(np.float32) +
            beta_refine * result_b_refine['denoised_image'].astype(np.float32) +
            gamma_refine * result_c_refine['denoised_image'].astype(np.float32)
        )
        
        # Blend base and refined results based on uncertainty mask
        blended_result = self._blend_refined_regions(
            base_denoised, refined_denoised, uncertain_mask
        )
        
        return {
            'refined_image': np.clip(blended_result, 0, 255).astype(np.uint8),
            'base_denoised': np.clip(base_denoised, 0, 255).astype(np.uint8),
            'refined_denoised': np.clip(refined_denoised, 0, 255).astype(np.uint8),
            'refinement_weights': (alpha_refine, beta_refine, gamma_refine),
            'base_weights': (alpha_base, beta_base, gamma_base),
            'uncertain_mask': uncertain_mask
        }
    
    def _blend_refined_regions(self, base_image, refined_image, uncertain_mask):
        """Smoothly blend refined regions with base denoising"""
        
        # Create smooth transition mask
        kernel_size = self.refinement_config['blend_kernel_size']
        kernel = cv2.getGaussianKernel(kernel_size, kernel_size // 3)
        kernel = kernel @ kernel.T
        
        # Smooth the uncertain mask for gradual blending
        if len(uncertain_mask.shape) == 2:
            smooth_mask = cv2.filter2D(uncertain_mask.astype(np.float32), -1, kernel)
        else:
            smooth_mask = uncertain_mask.astype(np.float32)
        
        # Ensure mask is in [0, 1] range
        smooth_mask = np.clip(smooth_mask, 0, 1)
        
        # Expand mask to match image dimensions if needed
        if len(base_image.shape) == 3 and len(smooth_mask.shape) == 2:
            smooth_mask = np.expand_dims(smooth_mask, axis=2)
            smooth_mask = np.repeat(smooth_mask, base_image.shape[2], axis=2)
        
        # Blend images
        blended = (1 - smooth_mask) * base_image + smooth_mask * refined_image
        
        return blended
    
    def iterative_refinement(self, image, noise_type, noise_level, 
                           threshold_percentile=80, max_iterations=3):
        """Apply iterative refinement until convergence"""
        
        current_image = image.copy()
        refinement_history = []
        
        for iteration in range(max_iterations):
            print(f"   🔄 Iteration {iteration + 1}/{max_iterations}")
            
            # Identify uncertain regions
            uncertainty_analysis = self.identify_uncertain_regions(
                current_image, noise_type, noise_level, threshold_percentile
            )
            
            uncertain_ratio = uncertainty_analysis['uncertain_pixel_ratio']
            print(f"      📊 Uncertain pixels: {uncertain_ratio*100:.1f}%")
            
            # If very few uncertain pixels, stop refinement
            if uncertain_ratio < 0.01:  # Less than 1%
                print(f"      ✅ Convergence achieved (< 1% uncertain pixels)")
                break
            
            # Apply refinement
            intensity_multiplier = 1.2 + iteration * 0.3  # Gradually increase intensity
            refinement_result = self.apply_targeted_refinement(
                current_image, 
                uncertainty_analysis['uncertain_mask'],
                noise_type, 
                noise_level,
                intensity_multiplier
            )
            
            # Update current image
            previous_image = current_image.copy()
            current_image = refinement_result['refined_image']
            
            # Check convergence based on image difference
            image_diff = np.mean(np.abs(current_image.astype(np.float32) - 
                                      previous_image.astype(np.float32)))
            normalized_diff = image_diff / 255.0
            
            print(f"      📈 Image change: {normalized_diff:.4f}")
            
            # Store iteration history
            refinement_history.append({
                'iteration': iteration + 1,
                'uncertain_ratio': uncertain_ratio,
                'image_change': normalized_diff,
                'intensity_multiplier': intensity_multiplier,
                'uncertainty_threshold': uncertainty_analysis['threshold_value']
            })
            
            # Check convergence
            if normalized_diff < self.refinement_config['convergence_threshold']:
                print(f"      ✅ Convergence achieved (change < {self.refinement_config['convergence_threshold']})")
                break
        
        return {
            'final_image': current_image,
            'refinement_history': refinement_history,
            'total_iterations': len(refinement_history),
            'converged': normalized_diff < self.refinement_config['convergence_threshold'] if 'normalized_diff' in locals() else True
        }
    
    def optimize_refinement_parameters(self, noise_type, max_images=20):
        """Optimize refinement parameters for specific noise type"""
        
        print(f"\n🎯 OPTIMIZING REFINEMENT FOR {noise_type.upper()}")
        print(f"=" * 50)
        
        # Load test images
        image_pairs = self._load_test_images(noise_type, max_images)
        
        if len(image_pairs) == 0:
            print(f"   ❌ No test images found for {noise_type}")
            return None
        
        print(f"   📊 Testing on {len(image_pairs)} image pairs")
        
        # Test different threshold percentiles
        best_threshold = 80
        best_improvement = 0.0
        threshold_results = []
        
        for threshold in self.refinement_config['uncertainty_thresholds']:
            print(f"   🔍 Testing threshold: {threshold}%")
            
            improvements = []
            processing_times = []
            
            for pair in image_pairs[:10]:  # Use subset for threshold optimization
                try:
                    start_time = time.time()
                    
                    # Apply base denoising
                    if self.optimal_weights and noise_type in self.optimal_weights:
                        weights = self.optimal_weights[noise_type]
                        alpha, beta, gamma = weights['alpha'], weights['beta'], weights['gamma']
                    else:
                        alpha, beta, gamma = 0.33, 0.33, 0.34
                    
                    # Base denoising
                    result_a = self.core_methods.method_a_denoise(pair['noisy'], noise_type, pair['noise_level'])
                    result_b = self.core_methods.method_b_denoise(pair['noisy'], noise_type, pair['noise_level'])
                    result_c = self.core_methods.method_c_denoise(pair['noisy'], noise_type, pair['noise_level'])
                    
                    base_denoised = (
                        alpha * result_a['denoised_image'].astype(np.float32) +
                        beta * result_b['denoised_image'].astype(np.float32) +
                        gamma * result_c['denoised_image'].astype(np.float32)
                    )
                    base_denoised = np.clip(base_denoised, 0, 255).astype(np.uint8)
                    
                    # Apply refinement
                    refinement_result = self.iterative_refinement(
                        pair['noisy'], noise_type, pair['noise_level'], 
                        threshold_percentile=threshold, max_iterations=2
                    )
                    
                    refined_image = refinement_result['final_image']
                    
                    # Calculate improvement
                    base_mse = np.mean((pair['clean'].astype(np.float32) - base_denoised.astype(np.float32))**2)
                    refined_mse = np.mean((pair['clean'].astype(np.float32) - refined_image.astype(np.float32))**2)
                    
                    improvement = (base_mse - refined_mse) / base_mse if base_mse > 0 else 0.0
                    improvements.append(improvement)
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                except Exception as e:
                    print(f"      ⚠️  Error testing threshold {threshold}: {e}")
                    continue
            
            avg_improvement = np.mean(improvements) if improvements else 0.0
            avg_time = np.mean(processing_times) if processing_times else 0.0
            
            threshold_results.append({
                'threshold': threshold,
                'avg_improvement': avg_improvement,
                'avg_processing_time': avg_time,
                'num_valid_tests': len(improvements)
            })
            
            print(f"      📈 Avg improvement: {avg_improvement*100:.2f}%")
            print(f"      ⏱️  Avg time: {avg_time:.2f}s")
            
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                best_threshold = threshold
        
        print(f"\n   🏆 BEST THRESHOLD: {best_threshold}% (improvement: {best_improvement*100:.2f}%)")
        
        # Final validation with best threshold
        final_validation = self._validate_refinement_strategy(
            image_pairs, noise_type, best_threshold
        )
        
        result = {
            'noise_type': noise_type,
            'optimal_threshold': best_threshold,
            'expected_improvement': best_improvement,
            'threshold_test_results': threshold_results,
            'final_validation': final_validation,
            'test_set_size': len(image_pairs),
            'timestamp': datetime.now().isoformat()
        }
        
        self.refinement_results[noise_type] = result
        return result
    
    def _validate_refinement_strategy(self, image_pairs, noise_type, threshold):
        """Validate refinement strategy on full test set"""
        
        print(f"   🔬 Final validation with threshold {threshold}%...")
        
        improvements = []
        processing_times = []
        iteration_counts = []
        
        for pair in image_pairs:
            try:
                start_time = time.time()
                
                # Base denoising
                if self.optimal_weights and noise_type in self.optimal_weights:
                    weights = self.optimal_weights[noise_type]
                    alpha, beta, gamma = weights['alpha'], weights['beta'], weights['gamma']
                else:
                    alpha, beta, gamma = 0.33, 0.33, 0.34
                
                result_a = self.core_methods.method_a_denoise(pair['noisy'], noise_type, pair['noise_level'])
                result_b = self.core_methods.method_b_denoise(pair['noisy'], noise_type, pair['noise_level'])
                result_c = self.core_methods.method_c_denoise(pair['noisy'], noise_type, pair['noise_level'])
                
                base_denoised = (
                    alpha * result_a['denoised_image'].astype(np.float32) +
                    beta * result_b['denoised_image'].astype(np.float32) +
                    gamma * result_c['denoised_image'].astype(np.float32)
                )
                base_denoised = np.clip(base_denoised, 0, 255).astype(np.uint8)
                
                # Refinement
                refinement_result = self.iterative_refinement(
                    pair['noisy'], noise_type, pair['noise_level'], 
                    threshold_percentile=threshold
                )
                
                refined_image = refinement_result['final_image']
                
                # Calculate metrics
                base_mse = np.mean((pair['clean'].astype(np.float32) - base_denoised.astype(np.float32))**2)
                refined_mse = np.mean((pair['clean'].astype(np.float32) - refined_image.astype(np.float32))**2)
                
                improvement = (base_mse - refined_mse) / base_mse if base_mse > 0 else 0.0
                improvements.append(improvement)
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                iteration_counts.append(refinement_result['total_iterations'])
                
            except Exception as e:
                continue
        
        return {
            'avg_improvement': np.mean(improvements) if improvements else 0.0,
            'improvement_std': np.std(improvements) if improvements else 0.0,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0.0,
            'avg_iterations': np.mean(iteration_counts) if iteration_counts else 0.0,
            'success_rate': len(improvements) / len(image_pairs),
            'num_valid_tests': len(improvements)
        }
    
    def _load_test_images(self, noise_type, sample_size):
        """Load test images for refinement optimization"""
        
        clean_images = []
        for category in ['photography', 'synthetic']:
            category_path = self.dataset_dir / "clean_images" / category
            if category_path.exists():
                clean_images.extend(list(category_path.glob("*.png")))
        
        if len(clean_images) == 0:
            return []
        
        selected_clean = np.random.choice(clean_images, min(sample_size, len(clean_images)), replace=False)
        
        image_pairs = []
        noisy_dir = self.dataset_dir / "noisy_images" / noise_type
        
        for clean_path in selected_clean:
            base_name = clean_path.stem
            
            for noise_level in [0.15, 0.25]:  # Use moderate noise levels
                noisy_name = f"{base_name}_{noise_type}_{noise_level:.2f}.png"
                noisy_path = noisy_dir / noisy_name
                
                if noisy_path.exists():
                    try:
                        clean_img = cv2.imread(str(clean_path))
                        noisy_img = cv2.imread(str(noisy_path))
                        
                        if clean_img is not None and noisy_img is not None:
                            if clean_img.shape == noisy_img.shape:
                                image_pairs.append({
                                    'clean': clean_img,
                                    'noisy': noisy_img,
                                    'noise_level': noise_level
                                })
                    except:
                        continue
        
        return image_pairs
    
    def run_complete_refinement_optimization(self):
        """Run refinement optimization for all noise types"""
        
        print(f"🚀 STARTING REFINEMENT OPTIMIZATION")
        print(f"=" * 60)
        
        start_time = time.time()
        noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
        
        for noise_type in noise_types:
            try:
                result = self.optimize_refinement_parameters(noise_type)
                if result:
                    print(f"   ✅ {noise_type} refinement optimization completed")
                
                # Save intermediate results
                self._save_refinement_results()
                
            except Exception as e:
                print(f"   ❌ {noise_type} refinement optimization failed: {e}")
        
        # Generate final report
        total_time = time.time() - start_time
        report = self._generate_refinement_report(total_time)
        
        return report
    
    def _save_refinement_results(self):
        """Save refinement optimization results"""
        results_path = self.experiment_dir / "refinement_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.refinement_results, f, indent=2, default=str)
    
    def _generate_refinement_report(self, total_time):
        """Generate comprehensive refinement optimization report"""
        
        report = {
            'refinement_summary': {
                'total_processing_time': total_time,
                'noise_types_processed': len(self.refinement_results),
                'successful_optimizations': len([r for r in self.refinement_results.values() if 'optimal_threshold' in r]),
                'timestamp': datetime.now().isoformat()
            },
            'optimal_thresholds': {},
            'expected_improvements': {},
            'detailed_results': self.refinement_results
        }
        
        # Extract optimal parameters
        for noise_type, result in self.refinement_results.items():
            if 'optimal_threshold' in result:
                report['optimal_thresholds'][noise_type] = result['optimal_threshold']
                report['expected_improvements'][noise_type] = result['expected_improvement']
        
        # Save report
        report_path = self.experiment_dir / "refinement_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n🎯 REFINEMENT OPTIMIZATION COMPLETE!")
        print(f"=" * 50)
        print(f"⏱️  Total Time: {total_time/60.0:.1f} minutes")
        
        print(f"\n🏆 OPTIMAL REFINEMENT THRESHOLDS:")
        for noise_type, threshold in report['optimal_thresholds'].items():
            improvement = report['expected_improvements'][noise_type]
            print(f"   {noise_type}: {threshold}% (improvement: {improvement*100:.2f}%)")
        
        print(f"\n📁 Report saved: {report_path}")
        
        return report

def main():
    """Execute refinement strategy optimization"""
    strategy = RefinementStrategy()
    report = strategy.run_complete_refinement_optimization()
    return report

if __name__ == "__main__":
    main()
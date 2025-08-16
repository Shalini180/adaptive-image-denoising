"""
Uncertainty Quantification System
Phase 1.4: Develop uncertainty indicators for adaptive refinement

Objectives:
- Design uncertainty measures that correlate with denoising errors
- Optimize uncertainty indicator weights for each noise type
- Validate uncertainty predictions against actual performance
- Enable targeted refinement of uncertain regions
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
from scipy import ndimage, stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import our core methods
from core_methods import CoreDenoisingMethods

class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification for adaptive denoising
    Develops indicators that predict denoising difficulty and guide refinement
    """
    
    def __init__(self, dataset_dir="../dataset", experiment_dir="uncertainty_quantification"):
        self.dataset_dir = Path(dataset_dir)
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Initialize core methods
        self.core_methods = CoreDenoisingMethods()
        
        # Load optimal weights from previous optimization
        self.optimal_weights = self._load_optimal_weights()
        
        # Uncertainty indicators configuration
        self.uncertainty_indicators = {
            'local_variance': {
                'description': 'Local noise variance estimation',
                'window_size': 5,
                'weight': 0.25
            },
            'edge_proximity': {
                'description': 'Distance to nearest edge features',
                'edge_threshold': 100,
                'weight': 0.25
            },
            'method_disagreement': {
                'description': 'Variance between different method outputs',
                'weight': 0.25
            },
            'snr_uncertainty': {
                'description': 'Local signal-to-noise ratio estimation',
                'weight': 0.25
            }
        }
        
        # Optimization results storage
        self.uncertainty_results = {}
        
        print(f"🔍 UNCERTAINTY QUANTIFICATION SYSTEM")
        print(f"=" * 50)
        print(f"📁 Dataset: {self.dataset_dir}")
        print(f"🧪 Experiment Dir: {self.experiment_dir}")
        print(f"📊 Uncertainty Indicators: {len(self.uncertainty_indicators)}")
        
        if self.optimal_weights:
            print(f"✅ Loaded optimal weights for {len(self.optimal_weights)} noise types")
        else:
            print(f"⚠️  No optimal weights found - will use default weights")
    
    def _load_optimal_weights(self):
        """Load optimal weights from weight optimization results"""
        weight_results_path = Path("weight_optimization/optimization_results.json")
        
        if weight_results_path.exists():
            try:
                with open(weight_results_path, 'r') as f:
                    results = json.load(f)
                
                optimal_weights = {}
                for noise_type, data in results.items():
                    if 'optimal_weights' in data:
                        optimal_weights[noise_type] = data['optimal_weights']
                
                return optimal_weights
                
            except Exception as e:
                print(f"   ⚠️  Error loading optimal weights: {e}")
                return None
        else:
            print(f"   ⚠️  Optimal weights file not found: {weight_results_path}")
            return None
    
    def calculate_local_variance(self, image, window_size=5):
        """Calculate local variance as uncertainty indicator"""
        
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate local variance using a sliding window
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        # Local mean
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Local variance = E[X²] - E[X]²
        local_mean_sq = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        local_variance = local_mean_sq - local_mean**2
        
        # Normalize to [0, 1]
        if np.max(local_variance) > 0:
            local_variance = local_variance / np.max(local_variance)
        
        return local_variance
    
    def calculate_edge_proximity(self, image, edge_threshold=100):
        """Calculate distance to nearest edge as uncertainty indicator"""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect edges using Canny
        edges = cv2.Canny(gray, edge_threshold//2, edge_threshold)
        
        # Calculate distance transform
        edge_distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        
        # Invert so that proximity to edges = high uncertainty
        max_distance = np.max(edge_distance)
        if max_distance > 0:
            edge_proximity = 1.0 - (edge_distance / max_distance)
        else:
            edge_proximity = np.ones_like(edge_distance)
        
        return edge_proximity
    
    def calculate_method_disagreement(self, image, noise_type, noise_level):
        """Calculate disagreement between different methods as uncertainty indicator"""
        
        try:
            # Apply all three methods
            result_a = self.core_methods.method_a_denoise(image, noise_type, noise_level)
            result_b = self.core_methods.method_b_denoise(image, noise_type, noise_level)
            result_c = self.core_methods.method_c_denoise(image, noise_type, noise_level)
            
            # Extract denoised images
            denoised_a = result_a['denoised_image'].astype(np.float32)
            denoised_b = result_b['denoised_image'].astype(np.float32)
            denoised_c = result_c['denoised_image'].astype(np.float32)
            
            # Calculate pixel-wise variance across methods
            if len(denoised_a.shape) == 3:
                # For color images, calculate variance across channels and methods
                method_stack = np.stack([denoised_a, denoised_b, denoised_c], axis=-1)
                method_disagreement = np.var(method_stack, axis=-1)
                # Average across color channels
                method_disagreement = np.mean(method_disagreement, axis=-1)
            else:
                # For grayscale images
                method_stack = np.stack([denoised_a, denoised_b, denoised_c], axis=-1)
                method_disagreement = np.var(method_stack, axis=-1)
            
            # Normalize to [0, 1]
            if np.max(method_disagreement) > 0:
                method_disagreement = method_disagreement / np.max(method_disagreement)
            
            return method_disagreement
            
        except Exception as e:
            print(f"   ⚠️  Error calculating method disagreement: {e}")
            # Return moderate uncertainty map
            if len(image.shape) == 3:
                return np.ones((image.shape[0], image.shape[1])) * 0.5
            else:
                return np.ones_like(image) * 0.5
    
    def calculate_snr_uncertainty(self, image, clean_image=None):
        """Calculate local signal-to-noise ratio uncertainty"""
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Estimate local signal and noise
        # Signal: local mean
        kernel_size = 7
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        local_signal = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Noise: high-frequency content (difference from local mean)
        noise_estimate = np.abs(gray.astype(np.float32) - local_signal)
        
        # Local noise variance
        local_noise_var = cv2.filter2D(noise_estimate**2, -1, kernel)
        
        # SNR estimation: signal / noise
        snr = local_signal / (np.sqrt(local_noise_var) + 1e-8)  # Add small epsilon
        
        # Convert SNR to uncertainty (lower SNR = higher uncertainty)
        max_snr = np.percentile(snr, 95)  # Use 95th percentile to avoid outliers
        snr_uncertainty = 1.0 - np.clip(snr / max_snr, 0, 1)
        
        return snr_uncertainty
    
    def compute_uncertainty_map(self, image, noise_type, noise_level, weights=None):
        """Compute combined uncertainty map using all indicators"""
        
        if weights is None:
            weights = [indicator['weight'] for indicator in self.uncertainty_indicators.values()]
        
        # Calculate individual uncertainty indicators
        local_var = self.calculate_local_variance(
            image, 
            self.uncertainty_indicators['local_variance']['window_size']
        )
        
        edge_prox = self.calculate_edge_proximity(
            image,
            self.uncertainty_indicators['edge_proximity']['edge_threshold']
        )
        
        method_disagree = self.calculate_method_disagreement(image, noise_type, noise_level)
        
        snr_uncert = self.calculate_snr_uncertainty(image)
        
        # Ensure all maps have the same shape (2D)
        if len(image.shape) == 3:
            target_shape = (image.shape[0], image.shape[1])
        else:
            target_shape = image.shape
        
        # Resize maps if necessary
        if local_var.shape != target_shape:
            local_var = cv2.resize(local_var, (target_shape[1], target_shape[0]))
        if edge_prox.shape != target_shape:
            edge_prox = cv2.resize(edge_prox, (target_shape[1], target_shape[0]))
        if method_disagree.shape != target_shape:
            method_disagree = cv2.resize(method_disagree, (target_shape[1], target_shape[0]))
        if snr_uncert.shape != target_shape:
            snr_uncert = cv2.resize(snr_uncert, (target_shape[1], target_shape[0]))
        
        # Weighted combination
        uncertainty_map = (
            weights[0] * local_var +
            weights[1] * edge_prox +
            weights[2] * method_disagree +
            weights[3] * snr_uncert
        )
        
        # Normalize to [0, 1]
        uncertainty_map = (uncertainty_map - np.min(uncertainty_map)) / (np.max(uncertainty_map) - np.min(uncertainty_map) + 1e-8)
        
        return {
            'combined_uncertainty': uncertainty_map,
            'individual_indicators': {
                'local_variance': local_var,
                'edge_proximity': edge_prox,
                'method_disagreement': method_disagree,
                'snr_uncertainty': snr_uncert
            },
            'weights_used': weights
        }
    
    def validate_uncertainty_prediction(self, image_pairs, noise_type, weights):
        """Validate uncertainty predictions against actual denoising errors"""
        
        correlations = []
        mse_errors = []
        
        for pair in tqdm(image_pairs, desc=f"Validating {noise_type}"):
            try:
                clean_img = pair['clean']
                noisy_img = pair['noisy']
                noise_level = pair['noise_level']
                
                # Compute uncertainty map
                uncertainty_result = self.compute_uncertainty_map(noisy_img, noise_type, noise_level, weights)
                uncertainty_map = uncertainty_result['combined_uncertainty']
                
                # Apply denoising with optimal weights
                if noise_type in self.optimal_weights:
                    opt_weights = self.optimal_weights[noise_type]
                    alpha, beta, gamma = opt_weights['alpha'], opt_weights['beta'], opt_weights['gamma']
                else:
                    # Default weights
                    alpha, beta, gamma = 0.33, 0.33, 0.34
                
                # Apply weighted combination
                result_a = self.core_methods.method_a_denoise(noisy_img, noise_type, noise_level)
                result_b = self.core_methods.method_b_denoise(noisy_img, noise_type, noise_level)
                result_c = self.core_methods.method_c_denoise(noisy_img, noise_type, noise_level)
                
                denoised = (
                    alpha * result_a['denoised_image'].astype(np.float32) +
                    beta * result_b['denoised_image'].astype(np.float32) +
                    gamma * result_c['denoised_image'].astype(np.float32)
                )
                denoised = np.clip(denoised, 0, 255).astype(np.uint8)
                
                # Calculate actual error map
                if len(clean_img.shape) == 3 and len(denoised.shape) == 3:
                    actual_error = np.mean(np.abs(clean_img.astype(np.float32) - denoised.astype(np.float32)), axis=2)
                elif len(clean_img.shape) == 2 and len(denoised.shape) == 2:
                    actual_error = np.abs(clean_img.astype(np.float32) - denoised.astype(np.float32))
                else:
                    # Handle shape mismatch
                    if len(clean_img.shape) == 3:
                        clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
                    else:
                        clean_gray = clean_img
                    
                    if len(denoised.shape) == 3:
                        denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
                    else:
                        denoised_gray = denoised
                    
                    actual_error = np.abs(clean_gray.astype(np.float32) - denoised_gray.astype(np.float32))
                
                # Normalize error map
                if np.max(actual_error) > 0:
                    actual_error = actual_error / np.max(actual_error)
                
                # Ensure same shape
                if uncertainty_map.shape != actual_error.shape:
                    uncertainty_map = cv2.resize(uncertainty_map, (actual_error.shape[1], actual_error.shape[0]))
                
                # Calculate correlation between uncertainty and actual error
                uncertainty_flat = uncertainty_map.flatten()
                error_flat = actual_error.flatten()
                
                if len(uncertainty_flat) > 1 and len(error_flat) > 1:
                    correlation = np.corrcoef(uncertainty_flat, error_flat)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
                
                # Calculate MSE between uncertainty and error
                mse = mean_squared_error(uncertainty_flat, error_flat)
                mse_errors.append(mse)
                
            except Exception as e:
                print(f"   ⚠️  Validation error: {e}")
                continue
        
        # Calculate validation metrics
        if correlations:
            avg_correlation = np.mean(correlations)
            std_correlation = np.std(correlations)
        else:
            avg_correlation = 0.0
            std_correlation = 0.0
        
        avg_mse = np.mean(mse_errors) if mse_errors else float('inf')
        
        return {
            'average_correlation': avg_correlation,
            'correlation_std': std_correlation,
            'average_mse': avg_mse,
            'num_valid_evaluations': len(correlations)
        }
    
    def optimize_uncertainty_weights(self, noise_type, max_images=30):
        """Optimize uncertainty indicator weights for specific noise type"""
        
        print(f"\n🎯 OPTIMIZING UNCERTAINTY WEIGHTS FOR {noise_type.upper()}")
        print(f"=" * 50)
        
        # Load test images
        image_pairs = self._load_test_images(noise_type, max_images)
        
        # Split into optimization and validation sets
        opt_pairs, val_pairs = train_test_split(image_pairs, test_size=0.3, random_state=42)
        
        print(f"   📊 Optimization set: {len(opt_pairs)} pairs")
        print(f"   📊 Validation set: {len(val_pairs)} pairs")
        
        # Generate weight combinations for uncertainty indicators
        weight_combinations = self._generate_uncertainty_weight_combinations()
        
        print(f"   🔄 Testing {len(weight_combinations)} weight combinations...")
        
        best_correlation = -1.0
        best_weights = None
        best_details = None
        
        for weights in tqdm(weight_combinations, desc=f"Optimizing uncertainty {noise_type}"):
            
            # Validate this weight combination
            validation_result = self.validate_uncertainty_prediction(opt_pairs, noise_type, weights)
            
            # Use correlation as optimization metric
            correlation = validation_result['average_correlation']
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_weights = weights
                best_details = validation_result
        
        print(f"\n   🏆 BEST UNCERTAINTY WEIGHTS FOR {noise_type}:")
        print(f"   Local Variance: {best_weights[0]:.2f}")
        print(f"   Edge Proximity: {best_weights[1]:.2f}")
        print(f"   Method Disagreement: {best_weights[2]:.2f}")
        print(f"   SNR Uncertainty: {best_weights[3]:.2f}")
        print(f"   Correlation: {best_correlation:.4f}")
        
        # Final validation on validation set
        final_validation = self.validate_uncertainty_prediction(val_pairs, noise_type, best_weights)
        
        result = {
            'noise_type': noise_type,
            'optimal_weights': {
                'local_variance': best_weights[0],
                'edge_proximity': best_weights[1],
                'method_disagreement': best_weights[2],
                'snr_uncertainty': best_weights[3]
            },
            'optimization_correlation': best_correlation,
            'optimization_details': best_details,
            'validation_results': final_validation,
            'optimization_set_size': len(opt_pairs),
            'validation_set_size': len(val_pairs),
            'timestamp': datetime.now().isoformat()
        }
        
        self.uncertainty_results[noise_type] = result
        return result
    
    def _generate_uncertainty_weight_combinations(self):
        """Generate weight combinations for uncertainty indicators"""
        
        combinations = []
        step = 0.1
        
        # Generate combinations that sum to 1.0
        for w1 in np.arange(0.1, 0.8, step):
            for w2 in np.arange(0.1, 0.8, step):
                for w3 in np.arange(0.1, 0.8, step):
                    w4 = 1.0 - w1 - w2 - w3
                    
                    if 0.1 <= w4 <= 0.8:
                        combinations.append([
                            round(w1, 1),
                            round(w2, 1), 
                            round(w3, 1),
                            round(w4, 1)
                        ])
        
        return combinations
    
    def _load_test_images(self, noise_type, sample_size):
        """Load test images for uncertainty optimization"""
        
        # Find clean images
        clean_images = []
        for category in ['photography', 'synthetic']:
            category_path = self.dataset_dir / "clean_images" / category
            if category_path.exists():
                clean_images.extend(list(category_path.glob("*.png")))
        
        if len(clean_images) == 0:
            raise ValueError("No clean images found")
        
        # Sample and load pairs
        selected_clean = np.random.choice(clean_images, min(sample_size, len(clean_images)), replace=False)
        
        image_pairs = []
        noisy_dir = self.dataset_dir / "noisy_images" / noise_type
        
        for clean_path in selected_clean:
            base_name = clean_path.stem
            
            for noise_level in [0.10, 0.20, 0.30]:  # Use subset of noise levels
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
                    except Exception as e:
                        continue
        
        return image_pairs
    
    def run_complete_uncertainty_optimization(self):
        """Run uncertainty quantification for all noise types"""
        
        print(f"🚀 STARTING UNCERTAINTY QUANTIFICATION")
        print(f"=" * 60)
        
        start_time = time.time()
        noise_types = ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson']
        
        for noise_type in noise_types:
            try:
                result = self.optimize_uncertainty_weights(noise_type)
                print(f"   ✅ {noise_type} uncertainty optimization completed")
                
                # Save intermediate results
                self._save_uncertainty_results()
                
            except Exception as e:
                print(f"   ❌ {noise_type} uncertainty optimization failed: {e}")
        
        # Generate final report
        total_time = time.time() - start_time
        report = self._generate_uncertainty_report(total_time)
        
        return report
    
    def _save_uncertainty_results(self):
        """Save uncertainty optimization results"""
        results_path = self.experiment_dir / "uncertainty_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.uncertainty_results, f, indent=2, default=str)
    
    def _generate_uncertainty_report(self, total_time):
        """Generate comprehensive uncertainty optimization report"""
        
        report = {
            'uncertainty_summary': {
                'total_processing_time': total_time,
                'noise_types_processed': len(self.uncertainty_results),
                'successful_optimizations': len([r for r in self.uncertainty_results.values() if 'optimal_weights' in r]),
                'timestamp': datetime.now().isoformat()
            },
            'optimal_uncertainty_weights': {},
            'correlation_performance': {},
            'detailed_results': self.uncertainty_results
        }
        
        # Extract optimal weights and performance
        for noise_type, result in self.uncertainty_results.items():
            if 'optimal_weights' in result:
                report['optimal_uncertainty_weights'][noise_type] = result['optimal_weights']
                
                if 'validation_results' in result:
                    report['correlation_performance'][noise_type] = {
                        'correlation': result['validation_results']['average_correlation'],
                        'mse': result['validation_results']['average_mse']
                    }
        
        # Save report
        report_path = self.experiment_dir / "uncertainty_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n🎯 UNCERTAINTY QUANTIFICATION COMPLETE!")
        print(f"=" * 50)
        print(f"⏱️  Total Time: {total_time/60.0:.1f} minutes")
        
        print(f"\n🏆 OPTIMAL UNCERTAINTY WEIGHTS:")
        for noise_type, weights in report['optimal_uncertainty_weights'].items():
            print(f"   {noise_type}:")
            for indicator, weight in weights.items():
                print(f"      {indicator}: {weight:.2f}")
        
        print(f"\n📊 CORRELATION PERFORMANCE:")
        for noise_type, perf in report['correlation_performance'].items():
            print(f"   {noise_type}: {perf['correlation']:.3f}")
        
        print(f"\n📁 Report saved: {report_path}")
        
        return report

def main():
    """Execute uncertainty quantification"""
    quantifier = UncertaintyQuantifier()
    report = quantifier.run_complete_uncertainty_optimization()
    return report

if __name__ == "__main__":
    main()
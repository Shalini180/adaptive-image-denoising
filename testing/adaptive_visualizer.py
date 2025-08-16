"""
Adaptive Process Visualizer
Detailed visualization of the adaptive denoising process showing:
- Step-by-step method selection
- Uncertainty map visualization
- Refinement iterations
- Comparison with other methods
- Performance metrics visualization
"""

import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'experiments'))

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import json

# Import our systems
from adaptive_denoiser import AdaptiveImageDenoiser
from comprehensive_tester import ComprehensiveTester

class AdaptiveProcessVisualizer:
    """
    Visualizes the complete adaptive denoising process with detailed analysis
    Shows decision making, uncertainty quantification, and refinement iterations
    """
    
    def __init__(self, output_dir="visualization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize systems
        self.adaptive_denoiser = AdaptiveImageDenoiser()
        self.tester = ComprehensiveTester()
        
        # Visualization configuration
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'adaptive': '#2E86AB',
            'bilateral': '#A23B72', 
            'nlm': '#F18F01',
            'gaussian': '#C73E1D',
            'median': '#592E83'
        }
        
        print(f"🎨 ADAPTIVE PROCESS VISUALIZER")
        print(f"=" * 50)
        print(f"📁 Output Directory: {self.output_dir}")
    
    def visualize_complete_process(self, clean_image, noise_type, noise_level, save_plots=True):
        """
        Complete visualization of adaptive denoising process
        Shows every step from noise detection to final result
        """
        
        print(f"\n🎨 VISUALIZING COMPLETE ADAPTIVE PROCESS")
        print(f"=" * 50)
        print(f"🎲 Noise: {noise_type} at level {noise_level}")
        
        # Add noise to image
        noisy_image = self.tester.add_specific_noise(clean_image, noise_type, noise_level)
        
        # Get detailed adaptive process results
        adaptive_result = self._get_detailed_adaptive_process(noisy_image)
        
        # Get comparison methods
        comparison_results = self.tester.apply_comparison_methods(noisy_image)
        
        # Create comprehensive visualization
        fig = self._create_complete_visualization(
            clean_image, noisy_image, adaptive_result, comparison_results, 
            noise_type, noise_level
        )
        
        if save_plots:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"adaptive_process_{noise_type}_{noise_level}_{timestamp}.png"
            filepath = self.output_dir / filename
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"📁 Visualization saved: {filepath}")
        
        # Show step-by-step analysis
        self._print_step_by_step_analysis(adaptive_result, noise_type, noise_level)
        
        return fig, adaptive_result
    
    def _get_detailed_adaptive_process(self, noisy_image):
        """Get detailed adaptive process with intermediate results"""
        
        # Apply our adaptive system
        result = self.adaptive_denoiser.denoise_image(noisy_image)
        
        # Get individual method results for detailed analysis
        noise_detection = result['metadata']['noise_detection']
        detected_type = noise_detection['primary_type']
        estimated_level = noise_detection['estimated_level']
        
        # Apply individual methods
        method_a_result = self.adaptive_denoiser.core_methods.method_a_denoise(
            noisy_image, detected_type, estimated_level
        )
        method_b_result = self.adaptive_denoiser.core_methods.method_b_denoise(
            noisy_image, detected_type, estimated_level
        )
        method_c_result = self.adaptive_denoiser.core_methods.method_c_denoise(
            noisy_image, detected_type, estimated_level
        )
        
        # Get optimal weights
        if detected_type in self.adaptive_denoiser.optimal_weights:
            weights = self.adaptive_denoiser.optimal_weights[detected_type]
            alpha, beta, gamma = weights['alpha'], weights['beta'], weights['gamma']
        else:
            alpha, beta, gamma = 0.33, 0.33, 0.34
        
        # Calculate weighted combination manually for visualization
        weighted_combination = (
            alpha * method_a_result['denoised_image'].astype(np.float32) +
            beta * method_b_result['denoised_image'].astype(np.float32) +
            gamma * method_c_result['denoised_image'].astype(np.float32)
        )
        weighted_combination = np.clip(weighted_combination, 0, 255).astype(np.uint8)
        
        # Calculate uncertainty map (simplified version)
        uncertainty_map = self._calculate_uncertainty_visualization(noisy_image, detected_type, estimated_level)
        
        # Detailed result structure
        detailed_result = {
            'original_result': result,
            'individual_methods': {
                'method_a': method_a_result,
                'method_b': method_b_result, 
                'method_c': method_c_result
            },
            'weighted_combination': weighted_combination,
            'optimal_weights': {'alpha': alpha, 'beta': beta, 'gamma': gamma},
            'noise_detection': noise_detection,
            'uncertainty_map': uncertainty_map,
            'processing_stages': result['processing_stages']
        }
        
        return detailed_result
    
    def _calculate_uncertainty_visualization(self, image, noise_type, noise_level):
        """Calculate uncertainty map for visualization"""
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Simple uncertainty indicators for visualization
        
        # 1. Local variance
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # 2. Edge proximity
        edges = cv2.Canny(gray, 50, 150)
        edge_distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        edge_proximity = 1.0 - (edge_distance / np.max(edge_distance)) if np.max(edge_distance) > 0 else np.ones_like(edge_distance)
        
        # 3. High frequency content
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        high_freq = (laplacian - np.min(laplacian)) / (np.max(laplacian) - np.min(laplacian)) if np.max(laplacian) > np.min(laplacian) else np.zeros_like(laplacian)
        
        # Combine uncertainty indicators (simplified weights)
        uncertainty = 0.4 * (local_variance / np.max(local_variance)) + 0.3 * edge_proximity + 0.3 * high_freq
        uncertainty = (uncertainty - np.min(uncertainty)) / (np.max(uncertainty) - np.min(uncertainty))
        
        return uncertainty
    
    def _create_complete_visualization(self, clean_image, noisy_image, adaptive_result, 
                                     comparison_results, noise_type, noise_level):
        """Create comprehensive visualization of the adaptive process"""
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Original images and noise detection
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_image(ax1, clean_image, "Clean Image")
        
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_image(ax2, noisy_image, f"Noisy Image\n({noise_type}, {noise_level})")
        
        ax3 = fig.add_subplot(gs[0, 2:4])
        self._plot_noise_detection_analysis(ax3, adaptive_result['noise_detection'])
        
        ax4 = fig.add_subplot(gs[0, 4])
        self._plot_image(ax4, adaptive_result['uncertainty_map'], "Uncertainty Map", cmap='hot')
        
        ax5 = fig.add_subplot(gs[0, 5])
        self._plot_image(ax5, adaptive_result['original_result']['final_image'], "Final Result")
        
        # Row 2: Individual methods
        ax6 = fig.add_subplot(gs[1, 0])
        self._plot_image(ax6, adaptive_result['individual_methods']['method_a']['denoised_image'], 
                        f"Method A\n(α={adaptive_result['optimal_weights']['alpha']:.2f})")
        
        ax7 = fig.add_subplot(gs[1, 1])
        self._plot_image(ax7, adaptive_result['individual_methods']['method_b']['denoised_image'], 
                        f"Method B\n(β={adaptive_result['optimal_weights']['beta']:.2f})")
        
        ax8 = fig.add_subplot(gs[1, 2])
        self._plot_image(ax8, adaptive_result['individual_methods']['method_c']['denoised_image'], 
                        f"Method C\n(γ={adaptive_result['optimal_weights']['gamma']:.2f})")
        
        ax9 = fig.add_subplot(gs[1, 3])
        self._plot_image(ax9, adaptive_result['weighted_combination'], "Weighted\nCombination")
        
        ax10 = fig.add_subplot(gs[1, 4:6])
        self._plot_weights_visualization(ax10, adaptive_result['optimal_weights'])
        
        # Row 3: Comparison methods
        methods = list(comparison_results.keys())[:5]  # Show first 5 methods
        for i, method_name in enumerate(methods):
            if i < 5:
                ax = fig.add_subplot(gs[2, i])
                method_result = comparison_results[method_name]
                self._plot_image(ax, method_result['denoised'], 
                               f"{method_result['method_name']}\n({method_result['processing_time']:.3f}s)")
        
        # Row 4: Performance metrics and analysis
        ax11 = fig.add_subplot(gs[3, :3])
        self._plot_performance_comparison(ax11, clean_image, adaptive_result, comparison_results)
        
        ax12 = fig.add_subplot(gs[3, 3:])
        self._plot_processing_timeline(ax12, adaptive_result['processing_stages'])
        
        # Add main title
        fig.suptitle(f'Adaptive Image Denoising Process Analysis\n'
                    f'Noise: {noise_type.upper()} (level {noise_level}) | '
                    f'Detected: {adaptive_result["noise_detection"]["primary_type"].upper()} '
                    f'(confidence: {adaptive_result["noise_detection"]["confidence"]:.3f})', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def _plot_image(self, ax, image, title, cmap=None):
        """Plot image with proper formatting"""
        
        if len(image.shape) == 3:
            # Color image
            if image.dtype == np.uint8:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(image)
        else:
            # Grayscale image
            ax.imshow(image, cmap=cmap or 'gray')
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    def _plot_noise_detection_analysis(self, ax, noise_detection):
        """Plot noise detection analysis"""
        
        scores = noise_detection['all_scores']
        noise_types = list(scores.keys())
        score_values = list(scores.values())
        detected_type = noise_detection['primary_type']
        
        # Create bar plot
        bars = ax.bar(noise_types, score_values, color=['red' if nt == detected_type else 'lightblue' for nt in noise_types])
        
        # Highlight detected noise type
        for i, (nt, bar) in enumerate(zip(noise_types, bars)):
            if nt == detected_type:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        ax.set_title(f'Noise Detection Scores\nDetected: {detected_type.upper()}', fontweight='bold')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 1.0)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add confidence text
        ax.text(0.02, 0.98, f'Confidence: {noise_detection["confidence"]:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_weights_visualization(self, ax, weights):
        """Plot method weights visualization"""
        
        methods = ['Method A\n(Adaptive Bilateral)', 'Method B\n(Multi-Method Consensus)', 'Method C\n(Edge-Preserving NLM)']
        weight_values = [weights['alpha'], weights['beta'], weights['gamma']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(weight_values, labels=methods, autopct='%1.2f', 
                                         colors=colors, startangle=90)
        
        ax.set_title('Optimal Method Weights\n(Empirically Optimized)', fontweight='bold')
        
        # Make text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_performance_comparison(self, ax, clean_image, adaptive_result, comparison_results):
        """Plot performance comparison metrics"""
        
        # Calculate metrics for all methods
        methods = ['Adaptive System']
        psnr_values = []
        ssim_values = []
        time_values = []
        
        # Adaptive system metrics
        adaptive_metrics = self.tester.calculate_comprehensive_metrics(
            clean_image, adaptive_result['original_result']['final_image'],
            adaptive_result['original_result']['metadata']['processing_time']
        )
        
        psnr_values.append(adaptive_metrics['psnr'])
        ssim_values.append(adaptive_metrics['ssim'])
        time_values.append(adaptive_metrics['processing_time'])
        
        # Comparison methods
        for method_name, method_result in comparison_results.items():
            methods.append(method_result['method_name'])
            
            method_metrics = self.tester.calculate_comprehensive_metrics(
                clean_image, method_result['denoised'], method_result['processing_time']
            )
            
            psnr_values.append(method_metrics['psnr'])
            ssim_values.append(method_metrics['ssim'])
            time_values.append(method_metrics['processing_time'])
        
        # Create subplot for metrics
        ax.clear()
        
        # PSNR comparison
        x = np.arange(len(methods))
        width = 0.25
        
        bars1 = ax.bar(x - width, psnr_values, width, label='PSNR (dB)', color='lightblue', alpha=0.8)
        bars2 = ax.bar(x, [s*40 for s in ssim_values], width, label='SSIM×40', color='lightgreen', alpha=0.8)
        bars3 = ax.bar(x + width, [1/t if t > 0 else 0 for t in time_values], width, label='Speed (1/time)', color='lightcoral', alpha=0.8)
        
        # Highlight adaptive system
        bars1[0].set_color('darkblue')
        bars2[0].set_color('darkgreen') 
        bars3[0].set_color('darkred')
        
        ax.set_title('Performance Comparison\n(Higher is Better for All Metrics)', fontweight='bold')
        ax.set_ylabel('Metric Value')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_processing_timeline(self, ax, processing_stages):
        """Plot processing timeline"""
        
        stage_names = [stage['stage'].replace('_', ' ').title() for stage in processing_stages]
        stage_times = [stage['processing_time'] for stage in processing_stages]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(stage_names))
        bars = ax.barh(y_pos, stage_times, color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'][:len(stage_names)])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stage_names)
        ax.set_xlabel('Processing Time (seconds)')
        ax.set_title('Processing Timeline\n(Adaptive System Stages)', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add time labels on bars
        for i, (bar, time_val) in enumerate(zip(bars, stage_times)):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                   f'{time_val:.3f}s', ha='left', va='center', fontsize=9)
        
        # Add total time
        total_time = sum(stage_times)
        ax.text(0.02, 0.98, f'Total: {total_time:.3f}s', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    def _print_step_by_step_analysis(self, adaptive_result, noise_type, noise_level):
        """Print detailed step-by-step analysis"""
        
        print(f"\n🔍 STEP-BY-STEP ADAPTIVE PROCESS ANALYSIS")
        print(f"=" * 60)
        
        # Step 1: Noise Detection
        noise_detection = adaptive_result['noise_detection']
        print(f"\n📊 STEP 1: NOISE DETECTION")
        print(f"   Input Noise: {noise_type} (level {noise_level})")
        print(f"   Detected Noise: {noise_detection['primary_type']}")
        print(f"   Detection Confidence: {noise_detection['confidence']:.3f}")
        print(f"   Estimated Level: {noise_detection['estimated_level']:.3f}")
        print(f"   ✅ Detection {'CORRECT' if noise_detection['primary_type'] == noise_type else 'INCORRECT'}")
        
        # Step 2: Method Selection and Weights
        weights = adaptive_result['optimal_weights']
        print(f"\n⚖️  STEP 2: EMPIRICAL WEIGHT SELECTION")
        print(f"   Selected for {noise_detection['primary_type']} noise:")
        print(f"   • Method A (Adaptive Bilateral): α = {weights['alpha']:.3f}")
        print(f"   • Method B (Multi-Method Consensus): β = {weights['beta']:.3f}")
        print(f"   • Method C (Edge-Preserving NLM): γ = {weights['gamma']:.3f}")
        print(f"   ✅ Weights sum to: {sum(weights.values()):.3f}")
        
        # Step 3: Individual Method Application
        individual_methods = adaptive_result['individual_methods']
        print(f"\n🔧 STEP 3: INDIVIDUAL METHOD APPLICATION")
        for method_name, method_result in individual_methods.items():
            method_full_name = {
                'method_a': 'Adaptive Bilateral Filter',
                'method_b': 'Multi-Method Consensus',
                'method_c': 'Edge-Preserving Non-Local Means'
            }[method_name]
            print(f"   {method_full_name}:")
            print(f"      Processing Time: {method_result['processing_time']:.3f}s")
            print(f"      Parameters Used: {method_result.get('parameters_used', 'Auto-adapted')}")
        
        # Step 4: Weighted Combination
        print(f"\n🎯 STEP 4: WEIGHTED COMBINATION")
        print(f"   Formula: Result = α×A + β×B + γ×C")
        print(f"   α×Method_A + β×Method_B + γ×Method_C")
        print(f"   {weights['alpha']:.3f}×A + {weights['beta']:.3f}×B + {weights['gamma']:.3f}×C")
        
        # Step 5: Uncertainty Analysis
        original_result = adaptive_result['original_result']
        if 'uncertainty_analysis' in original_result['metadata']:
            uncertainty = original_result['metadata']['uncertainty_analysis']
            print(f"\n🎯 STEP 5: UNCERTAINTY QUANTIFICATION")
            print(f"   Uncertain Pixel Ratio: {uncertainty['uncertain_pixel_ratio']*100:.1f}%")
            print(f"   Uncertainty Threshold: {uncertainty['threshold_percentile']}th percentile")
            print(f"   Weights Optimized: {uncertainty['weights_optimized']}")
        
        # Step 6: Refinement (if applied)
        if original_result['metadata']['refinement_applied']:
            print(f"\n🔄 STEP 6: ADAPTIVE REFINEMENT")
            print(f"   Refinement Applied: ✅ YES")
            refinement_stage = next((s for s in original_result['processing_stages'] if s['stage'] == 'refinement'), None)
            if refinement_stage:
                print(f"   Refinement Time: {refinement_stage['processing_time']:.3f}s")
                print(f"   Pixels Processed: {refinement_stage['results']['uncertain_pixels_processed']:,}")
        else:
            print(f"\n🔄 STEP 6: ADAPTIVE REFINEMENT")
            print(f"   Refinement Applied: ❌ NO (uncertainty below threshold)")
        
        # Step 7: Final Performance
        processing_stages = adaptive_result['processing_stages']
        total_time = sum(stage['processing_time'] for stage in processing_stages)
        print(f"\n📈 STEP 7: FINAL PERFORMANCE")
        print(f"   Total Processing Time: {total_time:.3f}s")
        print(f"   Number of Stages: {len(processing_stages)}")
        print(f"   Average Stage Time: {total_time/len(processing_stages):.3f}s")
        
        # Performance breakdown
        print(f"\n⏱️  DETAILED TIMING BREAKDOWN:")
        for stage in processing_stages:
            percentage = (stage['processing_time'] / total_time) * 100
            print(f"   {stage['stage'].replace('_', ' ').title()}: {stage['processing_time']:.3f}s ({percentage:.1f}%)")
    
    def visualize_refinement_iterations(self, clean_image, noise_type, noise_level, max_iterations=3):
        """Visualize refinement iterations step by step"""
        
        print(f"\n🔄 VISUALIZING REFINEMENT ITERATIONS")
        print(f"=" * 50)
        
        # Add noise
        noisy_image = self.tester.add_specific_noise(clean_image, noise_type, noise_level)
        
        # Simulate refinement iterations (simplified)
        current_image = noisy_image.copy()
        iteration_results = []
        
        for iteration in range(max_iterations):
            print(f"   Iteration {iteration + 1}...")
            
            # Apply denoising
            result = self.adaptive_denoiser.denoise_image(current_image)
            
            # Calculate metrics
            metrics = self.tester.calculate_comprehensive_metrics(clean_image, result['final_image'])
            
            # Calculate uncertainty
            uncertainty_map = self._calculate_uncertainty_visualization(current_image, noise_type, noise_level)
            uncertain_ratio = np.mean(uncertainty_map > 0.8)  # 80th percentile threshold
            
            iteration_results.append({
                'iteration': iteration + 1,
                'image': result['final_image'].copy(),
                'uncertainty_map': uncertainty_map,
                'uncertain_ratio': uncertain_ratio,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'processing_time': result['metadata']['processing_time']
            })
            
            # Update current image for next iteration
            current_image = result['final_image']
            
            # Stop if convergence achieved
            if uncertain_ratio < 0.05:  # Less than 5% uncertain pixels
                print(f"      Convergence achieved at iteration {iteration + 1}")
                break
        
        # Create refinement visualization
        fig = self._create_refinement_visualization(clean_image, noisy_image, iteration_results, noise_type, noise_level)
        
        # Save visualization
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"refinement_iterations_{noise_type}_{noise_level}_{timestamp}.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"📁 Refinement visualization saved: {filepath}")
        
        return fig, iteration_results
    
    def _create_refinement_visualization(self, clean_image, noisy_image, iteration_results, noise_type, noise_level):
        """Create visualization of refinement iterations"""
        
        num_iterations = len(iteration_results)
        
        # Create figure
        fig, axes = plt.subplots(3, num_iterations + 2, figsize=(4*(num_iterations + 2), 12))
        
        # First column: clean and noisy images
        self._plot_image(axes[0, 0], clean_image, "Clean Image")
        self._plot_image(axes[1, 0], noisy_image, f"Noisy Image\n({noise_type}, {noise_level})")
        axes[2, 0].axis('off')
        
        # Plot metrics evolution
        iterations = [r['iteration'] for r in iteration_results]
        psnr_values = [r['psnr'] for r in iteration_results]
        ssim_values = [r['ssim'] for r in iteration_results]
        uncertain_ratios = [r['uncertain_ratio'] for r in iteration_results]
        
        axes[2, 0].plot(iterations, psnr_values, 'bo-', label='PSNR (dB)', linewidth=2)
        axes[2, 0].set_xlabel('Iteration')
        axes[2, 0].set_ylabel('PSNR (dB)', color='b')
        axes[2, 0].tick_params(axis='y', labelcolor='b')
        axes[2, 0].grid(True, alpha=0.3)
        
        ax2 = axes[2, 0].twinx()
        ax2.plot(iterations, [s*100 for s in ssim_values], 'ro-', label='SSIM×100', linewidth=2)
        ax2.set_ylabel('SSIM×100', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        axes[2, 0].set_title('Metrics Evolution')
        
        # Iteration columns
        for i, result in enumerate(iteration_results):
            col = i + 1
            
            # Denoised image
            self._plot_image(axes[0, col], result['image'], 
                           f"Iteration {result['iteration']}\nPSNR: {result['psnr']:.1f} dB")
            
            # Uncertainty map
            self._plot_image(axes[1, col], result['uncertainty_map'], 
                           f"Uncertainty Map\n{result['uncertain_ratio']*100:.1f}% uncertain", cmap='hot')
            
            # Metrics
            axes[2, col].bar(['PSNR', 'SSIM×100', 'Speed'], 
                           [result['psnr'], result['ssim']*100, 1/result['processing_time']], 
                           color=['blue', 'red', 'green'], alpha=0.7)
            axes[2, col].set_title(f"Iteration {result['iteration']} Metrics")
            axes[2, col].set_ylabel('Value')
            axes[2, col].grid(True, alpha=0.3)
        
        # Final comparison
        if num_iterations > 0:
            col = num_iterations + 1
            final_result = iteration_results[-1]
            
            # Final vs original
            self._plot_image(axes[0, col], final_result['image'], 
                           f"Final Result\nPSNR: {final_result['psnr']:.1f} dB")
            
            # Improvement visualization
            improvement_psnr = final_result['psnr'] - iteration_results[0]['psnr'] if len(iteration_results) > 1 else 0
            improvement_ssim = final_result['ssim'] - iteration_results[0]['ssim'] if len(iteration_results) > 1 else 0
            
            axes[1, col].bar(['PSNR\nImprovement', 'SSIM\nImprovement'], 
                           [improvement_psnr, improvement_ssim*100], 
                           color=['green' if improvement_psnr > 0 else 'red', 
                                 'green' if improvement_ssim > 0 else 'red'], alpha=0.7)
            axes[1, col].set_title('Refinement Improvement')
            axes[1, col].set_ylabel('Improvement')
            axes[1, col].grid(True, alpha=0.3)
            
            # Summary statistics
            axes[2, col].text(0.1, 0.9, f"Iterations: {num_iterations}", transform=axes[2, col].transAxes, fontsize=12)
            axes[2, col].text(0.1, 0.8, f"Final PSNR: {final_result['psnr']:.2f} dB", transform=axes[2, col].transAxes, fontsize=12)
            axes[2, col].text(0.1, 0.7, f"Final SSIM: {final_result['ssim']:.4f}", transform=axes[2, col].transAxes, fontsize=12)
            axes[2, col].text(0.1, 0.6, f"Total Time: {sum(r['processing_time'] for r in iteration_results):.3f}s", 
                             transform=axes[2, col].transAxes, fontsize=12)
            axes[2, col].set_title('Summary')
            axes[2, col].axis('off')
        
        plt.suptitle(f'Adaptive Refinement Iterations - {noise_type.upper()} Noise (level {noise_level})', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

def main():
    """Run adaptive process visualization"""
    
    print("🎨 ADAPTIVE PROCESS VISUALIZER")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = AdaptiveProcessVisualizer()
    
    # Create test image
    tester = ComprehensiveTester()
    test_image = tester._generate_natural_image(256, 256)
    
    # Visualize complete process
    print("\n1. Visualizing complete adaptive process...")
    fig1, adaptive_result = visualizer.visualize_complete_process(
        test_image, 'gaussian', 0.15, save_plots=True
    )
    
    # Visualize refinement iterations
    print("\n2. Visualizing refinement iterations...")
    fig2, refinement_results = visualizer.visualize_refinement_iterations(
        test_image, 'salt_pepper', 0.20, max_iterations=3
    )
    
    plt.show()
    
    print(f"\n✅ Visualization complete!")
    print(f"📁 Results saved to: {visualizer.output_dir}")

if __name__ == "__main__":
    main()
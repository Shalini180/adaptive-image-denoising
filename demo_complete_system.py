"""
Complete System Demonstration
Shows the full adaptive denoising system with comprehensive analysis

This demo will:
1. Create test images with different noise types
2. Apply our adaptive denoising system
3. Compare with state-of-the-art methods (OpenCV, scikit-image)
4. Calculate all metrics (PSNR, SSIM, processing time)
5. Show step-by-step adaptive process
6. Visualize uncertainty maps and refinement
7. Generate comprehensive reports

Run with: python demo_complete_system.py
"""

import numpy as np
import cv2
import sys
import os
sys.path.append('src')
sys.path.append('testing')

from pathlib import Path
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import our systems
from adaptive_denoiser import AdaptiveImageDenoiser
from testing.comprehensive_tester import ComprehensiveTester
from testing.adaptive_visualizer import AdaptiveProcessVisualizer
from testing.master_testing_suite import MasterTestingSuite

def create_demo_images():
    """Create a variety of test images for demonstration"""
    
    print("📸 Creating demonstration images...")
    
    # Create test image generator
    tester = ComprehensiveTester()
    
    # Generate different types of test images
    demo_images = []
    
    # 1. Natural-like image
    natural_img = tester._generate_natural_image(256, 256)
    demo_images.append(('natural', natural_img))
    
    # 2. Geometric shapes
    geometric_img = tester._generate_geometric_image(256, 256)
    demo_images.append(('geometric', geometric_img))
    
    # 3. Texture pattern
    texture_img = tester._generate_texture_image(256, 256)
    demo_images.append(('texture', texture_img))
    
    # 4. Load a real image if available
    sample_images = list(Path('.').glob('*.jpg')) + list(Path('.').glob('*.png'))
    if sample_images:
        try:
            real_img = cv2.imread(str(sample_images[0]))
            if real_img is not None:
                # Resize to consistent size
                real_img = cv2.resize(real_img, (256, 256))
                demo_images.append(('real_image', real_img))
        except:
            pass
    
    print(f"   ✅ Created {len(demo_images)} test images")
    return demo_images

def demonstrate_single_image_analysis(clean_image, image_name="demo"):
    """Demonstrate comprehensive analysis on a single image"""
    
    print(f"\n🔬 DETAILED ANALYSIS: {image_name.upper()}")
    print("=" * 50)
    
    # Initialize systems
    denoiser = AdaptiveImageDenoiser()
    tester = ComprehensiveTester()
    
    # Test different noise types
    noise_scenarios = [
        {'type': 'gaussian', 'level': 0.15},
        {'type': 'salt_pepper', 'level': 0.10},
        {'type': 'speckle', 'level': 0.20}
    ]
    
    for scenario in noise_scenarios:
        noise_type = scenario['type']
        noise_level = scenario['level']
        
        print(f"\n🎲 Testing {noise_type.upper()} noise (level {noise_level})")
        print("-" * 30)
        
        # Add noise to image
        noisy_image = tester.add_specific_noise(clean_image, noise_type, noise_level)
        
        # Apply our adaptive system
        print("🎯 Applying adaptive denoising system...")
        start_time = time.time()
        adaptive_result = denoiser.denoise_image(noisy_image)
        adaptive_time = time.time() - start_time
        
        # Apply comparison methods
        print("📊 Applying comparison methods...")
        comparison_results = tester.apply_comparison_methods(noisy_image)
        
        # Calculate comprehensive metrics
        print("📈 Calculating metrics...")
        
        # Our adaptive system metrics
        adaptive_metrics = tester.calculate_comprehensive_metrics(
            clean_image, adaptive_result['final_image'], adaptive_time
        )
        
        print(f"\n🎯 ADAPTIVE SYSTEM RESULTS:")
        print(f"   Detected Noise: {adaptive_result['metadata']['noise_detection']['primary_type']} "
              f"(confidence: {adaptive_result['metadata']['noise_detection']['confidence']:.3f})")
        print(f"   ✅ Detection {'CORRECT' if adaptive_result['metadata']['noise_detection']['primary_type'] == noise_type else 'INCORRECT'}")
        print(f"   PSNR: {adaptive_metrics['psnr']:.2f} dB")
        print(f"   SSIM: {adaptive_metrics['ssim']:.4f}")
        print(f"   Processing Time: {adaptive_metrics['processing_time']:.3f}s")
        print(f"   Refinement Applied: {'✅ YES' if adaptive_result['metadata']['refinement_applied'] else '❌ NO'}")
        
        # Show processing stages
        print(f"\n⏱️  PROCESSING BREAKDOWN:")
        for stage in adaptive_result['processing_stages']:
            print(f"   {stage['stage'].replace('_', ' ').title()}: {stage['processing_time']:.3f}s")
        
        # Comparison with other methods
        print(f"\n📊 COMPARISON WITH OTHER METHODS:")
        best_classical_psnr = 0
        best_classical_method = ""
        
        for method_name, method_result in comparison_results.items():
            method_metrics = tester.calculate_comprehensive_metrics(
                clean_image, method_result['denoised'], method_result['processing_time']
            )
            
            print(f"   {method_result['method_name']}:")
            print(f"      PSNR: {method_metrics['psnr']:.2f} dB, SSIM: {method_metrics['ssim']:.4f}, "
                  f"Time: {method_metrics['processing_time']:.3f}s")
            
            if method_metrics['psnr'] > best_classical_psnr:
                best_classical_psnr = method_metrics['psnr']
                best_classical_method = method_result['method_name']
        
        # Performance comparison
        psnr_improvement = ((adaptive_metrics['psnr'] - best_classical_psnr) / best_classical_psnr) * 100
        
        print(f"\n🏆 PERFORMANCE SUMMARY:")
        print(f"   Best Classical Method: {best_classical_method} ({best_classical_psnr:.2f} dB)")
        print(f"   Our Adaptive System: {adaptive_metrics['psnr']:.2f} dB")
        print(f"   PSNR Improvement: {psnr_improvement:+.2f}%")
        print(f"   Ranking: {'🥇 #1' if adaptive_metrics['psnr'] > best_classical_psnr else '🥈 #2+'}")
        
        # Show detailed adaptive process
        print(f"\n🔍 ADAPTIVE PROCESS DETAILS:")
        noise_detection = adaptive_result['metadata']['noise_detection']
        print(f"   All Detection Scores: {noise_detection['all_scores']}")
        
        if 'denoising_parameters' in adaptive_result['metadata']:
            weights = adaptive_result['metadata']['denoising_parameters']['method_weights']
            print(f"   Selected Weights: α={weights['alpha']:.3f}, β={weights['beta']:.3f}, γ={weights['gamma']:.3f}")
        
        if 'uncertainty_analysis' in adaptive_result['metadata']:
            uncertainty = adaptive_result['metadata']['uncertainty_analysis']
            print(f"   Uncertain Pixels: {uncertainty['uncertain_pixel_ratio']*100:.1f}%")

def demonstrate_batch_analysis():
    """Demonstrate batch processing with statistical analysis"""
    
    print(f"\n📊 BATCH ANALYSIS DEMONSTRATION")
    print("=" * 50)
    
    # Create multiple test images
    demo_images = create_demo_images()
    
    # Initialize systems
    denoiser = AdaptiveImageDenoiser()
    tester = ComprehensiveTester()
    
    # Collect results for statistical analysis
    all_results = []
    
    for image_name, clean_image in demo_images:
        print(f"\n🖼️  Processing {image_name}...")
        
        # Test with Gaussian noise at moderate level
        noise_type = 'gaussian'
        noise_level = 0.15
        
        # Add noise
        noisy_image = tester.add_specific_noise(clean_image, noise_type, noise_level)
        
        # Apply adaptive denoising
        adaptive_result = denoiser.denoise_image(noisy_image)
        
        # Calculate metrics
        metrics = tester.calculate_comprehensive_metrics(
            clean_image, adaptive_result['final_image'], 
            adaptive_result['metadata']['processing_time']
        )
        
        # Store results
        result_data = {
            'image_name': image_name,
            'metrics': metrics,
            'detection_correct': adaptive_result['metadata']['noise_detection']['primary_type'] == noise_type,
            'detection_confidence': adaptive_result['metadata']['noise_detection']['confidence'],
            'refinement_applied': adaptive_result['metadata']['refinement_applied']
        }
        
        all_results.append(result_data)
        
        print(f"   PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}, "
              f"Time: {metrics['processing_time']:.3f}s")
    
    # Calculate aggregate statistics
    print(f"\n📈 AGGREGATE STATISTICS:")
    avg_psnr = np.mean([r['metrics']['psnr'] for r in all_results])
    avg_ssim = np.mean([r['metrics']['ssim'] for r in all_results])
    avg_time = np.mean([r['metrics']['processing_time'] for r in all_results])
    detection_accuracy = np.mean([r['detection_correct'] for r in all_results])
    refinement_rate = np.mean([r['refinement_applied'] for r in all_results])
    
    print(f"   Average PSNR: {avg_psnr:.2f} ± {np.std([r['metrics']['psnr'] for r in all_results]):.2f} dB")
    print(f"   Average SSIM: {avg_ssim:.4f} ± {np.std([r['metrics']['ssim'] for r in all_results]):.4f}")
    print(f"   Average Time: {avg_time:.3f} ± {np.std([r['metrics']['processing_time'] for r in all_results]):.3f}s")
    print(f"   Detection Accuracy: {detection_accuracy*100:.1f}%")
    print(f"   Refinement Rate: {refinement_rate*100:.1f}%")

def demonstrate_visualization():
    """Demonstrate adaptive process visualization"""
    
    print(f"\n🎨 VISUALIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Create a test image
    tester = ComprehensiveTester()
    test_image = tester._generate_natural_image(256, 256)
    
    # Initialize visualizer
    visualizer = AdaptiveProcessVisualizer("demo_visualizations")
    
    # Create visualizations for different noise types
    visualization_scenarios = [
        {'noise_type': 'gaussian', 'noise_level': 0.15},
        {'noise_type': 'salt_pepper', 'noise_level': 0.10}
    ]
    
    for scenario in visualization_scenarios:
        noise_type = scenario['noise_type']
        noise_level = scenario['noise_level']
        
        print(f"   🎨 Creating visualization for {noise_type} noise...")
        
        try:
            # Complete process visualization
            fig, adaptive_result = visualizer.visualize_complete_process(
                test_image, noise_type, noise_level, save_plots=True
            )
            
            print(f"      ✅ Complete process visualization saved")
            
            # Refinement visualization
            fig2, refinement_results = visualizer.visualize_refinement_iterations(
                test_image, noise_type, noise_level, max_iterations=2
            )
            
            print(f"      ✅ Refinement visualization saved")
            
        except Exception as e:
            print(f"      ⚠️  Visualization error: {e}")
    
    print(f"   📁 Visualizations saved to: demo_visualizations/")

def run_quick_comprehensive_test():
    """Run a quick version of the comprehensive test suite"""
    
    print(f"\n🧪 QUICK COMPREHENSIVE TEST")
    print("=" * 50)
    
    # Initialize master testing suite with quick mode
    suite = MasterTestingSuite("demo_comprehensive_results")
    
    # Configure for quick testing
    suite.test_config['detailed_analysis_samples'] = 2
    suite.test_config['visualization_samples'] = 1
    
    print("🚀 Running quick comprehensive evaluation...")
    
    try:
        # Run comprehensive evaluation
        final_report = suite.run_complete_evaluation()
        
        # Show key results
        print(f"\n🎯 QUICK TEST RESULTS:")
        grade = final_report['system_performance_grade']
        print(f"   System Grade: {grade['letter_grade']} ({grade['numerical_score']}/100)")
        
        exec_summary = final_report['executive_summary']
        print(f"   Average PSNR: {exec_summary['system_performance']['average_psnr']:.2f} dB")
        print(f"   Average SSIM: {exec_summary['system_performance']['average_ssim']:.4f}")
        print(f"   Processing Time: {exec_summary['system_performance']['average_processing_time']:.3f}s")
        
        improvement = exec_summary['improvement_over_classical']
        print(f"   PSNR Improvement: {improvement['psnr_improvement_percentage']:+.2f}%")
        print(f"   Tests Outperformed: {improvement['tests_outperformed_percentage']:.1f}%")
        
        print(f"\n📁 Complete results saved to: demo_comprehensive_results/")
        
    except Exception as e:
        print(f"   ⚠️  Quick test error: {e}")

def main():
    """Main demonstration function"""
    
    print("🎯 COMPLETE ADAPTIVE DENOISING SYSTEM DEMONSTRATION")
    print("=" * 60)
    print(f"⏱️  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if system is ready
    try:
        from adaptive_denoiser import AdaptiveImageDenoiser
        denoiser = AdaptiveImageDenoiser()
        print("✅ Adaptive denoising system loaded successfully")
    except Exception as e:
        print(f"❌ Error loading system: {e}")
        return
    
    # Demo 1: Single image detailed analysis
    print(f"\n" + "="*60)
    print("DEMO 1: DETAILED SINGLE IMAGE ANALYSIS")
    print("="*60)
    
    # Create a demo image
    tester = ComprehensiveTester()
    demo_image = tester._generate_natural_image(256, 256)
    
    # Run detailed analysis
    demonstrate_single_image_analysis(demo_image, "natural_pattern")
    
    # Demo 2: Batch processing
    print(f"\n" + "="*60)
    print("DEMO 2: BATCH PROCESSING WITH STATISTICS")
    print("="*60)
    
    demonstrate_batch_analysis()
    
    # Demo 3: Visualization
    print(f"\n" + "="*60)
    print("DEMO 3: ADAPTIVE PROCESS VISUALIZATION")
    print("="*60)
    
    demonstrate_visualization()
    
    # Demo 4: Comprehensive test (quick version)
    print(f"\n" + "="*60)
    print("DEMO 4: COMPREHENSIVE EVALUATION (QUICK)")
    print("="*60)
    
    run_quick_comprehensive_test()
    
    # Summary
    print(f"\n🎉 DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("📁 Results and visualizations saved to:")
    print("   • demo_visualizations/ - Process visualizations")
    print("   • demo_comprehensive_results/ - Complete test results")
    print("   • testing_results/ - Individual test outputs")
    
    print(f"\n🚀 TO RUN FULL EVALUATION:")
    print("   python testing/master_testing_suite.py --full")
    
    print(f"\n🎨 TO CREATE MORE VISUALIZATIONS:")
    print("   python testing/adaptive_visualizer.py")
    
    print(f"\n📊 TO RUN COMPREHENSIVE TESTS:")
    print("   python testing/comprehensive_tester.py")

if __name__ == "__main__":
    main()
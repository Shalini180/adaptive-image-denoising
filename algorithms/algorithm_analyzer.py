import pandas as pd
import numpy as np
import json
from datetime import datetime

class AlgorithmAnalyzer:
    """
    Comprehensive analysis framework for denoising algorithms
    Evaluates methods across multiple criteria for systematic selection
    """
    
    def __init__(self):
        self.algorithms = {
            'spatial_domain': {
                'gaussian_filter': {
                    'complexity': 'O(n²)',
                    'memory': 'O(1)',
                    'noise_types': ['gaussian'],
                    'edge_preservation': 'Poor',
                    'implementation_difficulty': 'Easy',
                    'parameters': ['sigma'],
                    'strengths': ['Fast', 'Simple', 'Stable'],
                    'weaknesses': ['Blurs edges', 'Poor texture preservation'],
                    'typical_psnr_gain': '1-2 dB',
                    'computational_score': 0.9,
                    'quality_score': 0.3,
                    'adaptability_score': 0.4
                },
                'median_filter': {
                    'complexity': 'O(n²log n)',
                    'memory': 'O(1)',
                    'noise_types': ['salt_pepper', 'impulse'],
                    'edge_preservation': 'Good',
                    'implementation_difficulty': 'Easy',
                    'parameters': ['kernel_size'],
                    'strengths': ['Excellent for impulse noise', 'Edge preserving'],
                    'weaknesses': ['Slow for large kernels', 'Poor for Gaussian noise'],
                    'typical_psnr_gain': '3-8 dB (salt-pepper)',
                    'computational_score': 0.6,
                    'quality_score': 0.7,
                    'adaptability_score': 0.3
                },
                'bilateral_filter': {
                    'complexity': 'O(n²)',
                    'memory': 'O(1)',
                    'noise_types': ['gaussian'],
                    'edge_preservation': 'Excellent',
                    'implementation_difficulty': 'Medium',
                    'parameters': ['sigma_space', 'sigma_intensity'],
                    'strengths': ['Edge preserving', 'Good for natural images', 'Tunable'],
                    'weaknesses': ['Parameter sensitive', 'Moderate complexity'],
                    'typical_psnr_gain': '2-4 dB',
                    'computational_score': 0.7,
                    'quality_score': 0.8,
                    'adaptability_score': 0.7
                },
                'adaptive_bilateral': {
                    'complexity': 'O(n²)',
                    'memory': 'O(1)',
                    'noise_types': ['gaussian', 'uniform'],
                    'edge_preservation': 'Excellent',
                    'implementation_difficulty': 'Medium',
                    'parameters': ['sigma_space', 'sigma_intensity', 'noise_variance'],
                    'strengths': ['Edge preserving', 'Noise-adaptive', 'Fast'],
                    'weaknesses': ['Requires noise estimation', 'Parameter tuning'],
                    'typical_psnr_gain': '3-5 dB',
                    'computational_score': 0.7,
                    'quality_score': 0.8,
                    'adaptability_score': 0.9
                }
            },
            'transform_domain': {
                'wiener_filter': {
                    'complexity': 'O(n²log n)',
                    'memory': 'O(n²)',
                    'noise_types': ['gaussian', 'uniform'],
                    'edge_preservation': 'Good',
                    'implementation_difficulty': 'Hard',
                    'parameters': ['noise_variance', 'signal_variance'],
                    'strengths': ['Optimal for known statistics', 'Good SNR improvement'],
                    'weaknesses': ['Requires noise/signal statistics', 'Complex implementation'],
                    'typical_psnr_gain': '2-4 dB',
                    'computational_score': 0.5,
                    'quality_score': 0.7,
                    'adaptability_score': 0.4
                },
                'bm3d': {
                    'complexity': 'O(n³)',
                    'memory': 'O(n²)',
                    'noise_types': ['gaussian'],
                    'edge_preservation': 'Excellent',
                    'implementation_difficulty': 'Very Hard',
                    'parameters': ['sigma', 'block_size', 'search_window'],
                    'strengths': ['State-of-art quality', 'Excellent texture preservation'],
                    'weaknesses': ['Very slow', 'Complex implementation', 'Memory intensive'],
                    'typical_psnr_gain': '3-6 dB',
                    'computational_score': 0.2,
                    'quality_score': 0.95,
                    'adaptability_score': 0.3
                },
                'bm3d_simplified': {
                    'complexity': 'O(n²log n)',
                    'memory': 'O(n²)',
                    'noise_types': ['gaussian'],
                    'edge_preservation': 'Good',
                    'implementation_difficulty': 'Hard',
                    'parameters': ['sigma', 'block_size'],
                    'strengths': ['Good quality', 'Faster than full BM3D'],
                    'weaknesses': ['Still complex', 'Memory requirements'],
                    'typical_psnr_gain': '2-4 dB',
                    'computational_score': 0.4,
                    'quality_score': 0.8,
                    'adaptability_score': 0.4
                }
            },
            'variational': {
                'total_variation': {
                    'complexity': 'O(n²k)',  # k=iterations
                    'memory': 'O(n²)',
                    'noise_types': ['gaussian', 'impulse'],
                    'edge_preservation': 'Excellent',
                    'implementation_difficulty': 'Hard',
                    'parameters': ['lambda', 'iterations'],
                    'strengths': ['Edge preserving', 'Good for piecewise smooth'],
                    'weaknesses': ['Staircasing artifacts', 'Parameter tuning needed'],
                    'typical_psnr_gain': '2-5 dB',
                    'computational_score': 0.3,
                    'quality_score': 0.8,
                    'adaptability_score': 0.5
                }
            },
            'statistical': {
                'non_local_means': {
                    'complexity': 'O(n⁴)',
                    'memory': 'O(n²)',
                    'noise_types': ['gaussian'],
                    'edge_preservation': 'Excellent',
                    'implementation_difficulty': 'Medium',
                    'parameters': ['h', 'template_window', 'search_window'],
                    'strengths': ['Excellent texture preservation', 'Good edge preservation'],
                    'weaknesses': ['Very slow', 'Parameter sensitive'],
                    'typical_psnr_gain': '2-4 dB',
                    'computational_score': 0.1,
                    'quality_score': 0.9,
                    'adaptability_score': 0.6
                },
                'edge_preserving_nlm': {
                    'complexity': 'O(n³)',
                    'memory': 'O(n²)',
                    'noise_types': ['gaussian', 'speckle'],
                    'edge_preservation': 'Excellent',
                    'implementation_difficulty': 'Medium',
                    'parameters': ['h', 'template_window', 'search_window', 'edge_threshold'],
                    'strengths': ['Superior edge preservation', 'Good texture preservation'],
                    'weaknesses': ['Slower than bilateral', 'Parameter tuning'],
                    'typical_psnr_gain': '3-5 dB',
                    'computational_score': 0.3,
                    'quality_score': 0.9,
                    'adaptability_score': 0.8
                },
                'lee_filter': {
                    'complexity': 'O(n²)',
                    'memory': 'O(1)',
                    'noise_types': ['speckle'],
                    'edge_preservation': 'Good',
                    'implementation_difficulty': 'Medium',
                    'parameters': ['window_size', 'variance_threshold'],
                    'strengths': ['Excellent for speckle', 'Adaptive', 'Fast'],
                    'weaknesses': ['Specific to speckle noise', 'Parameter dependent'],
                    'typical_psnr_gain': '3-6 dB (speckle)',
                    'computational_score': 0.8,
                    'quality_score': 0.7,
                    'adaptability_score': 0.4
                }
            },
            'consensus_methods': {
                'multi_method_consensus': {
                    'complexity': 'O(n² × M)',  # M = number of base methods
                    'memory': 'O(n² × M)',
                    'noise_types': ['all'],
                    'edge_preservation': 'Good to Excellent',
                    'implementation_difficulty': 'Medium',
                    'parameters': ['base_methods', 'consensus_strategy', 'weights'],
                    'strengths': ['Robust across noise types', 'Combines method strengths'],
                    'weaknesses': ['Increased computation', 'Parameter complexity'],
                    'typical_psnr_gain': '2-5 dB',
                    'computational_score': 0.4,
                    'quality_score': 0.8,
                    'adaptability_score': 0.9
                },
                'weighted_median_consensus': {
                    'complexity': 'O(n² × M log M)',
                    'memory': 'O(n² × M)',
                    'noise_types': ['all'],
                    'edge_preservation': 'Excellent',
                    'implementation_difficulty': 'Medium',
                    'parameters': ['base_methods', 'weights', 'confidence_scores'],
                    'strengths': ['Robust to outliers', 'Edge preserving'],
                    'weaknesses': ['Computational overhead', 'Complex weighting'],
                    'typical_psnr_gain': '3-6 dB',
                    'computational_score': 0.3,
                    'quality_score': 0.85,
                    'adaptability_score': 0.9
                }
            }
        }
    
    def get_all_algorithms(self):
        """Get flattened list of all algorithms with their categories"""
        all_algs = []
        for category, algs in self.algorithms.items():
            for alg_name, alg_data in algs.items():
                alg_data_copy = alg_data.copy()
                alg_data_copy['name'] = alg_name
                alg_data_copy['category'] = category
                all_algs.append(alg_data_copy)
        return all_algs
    
    def analyze_compatibility(self, alg1_name, alg2_name, alg3_name):
        """Analyze how well three algorithms complement each other"""
        
        # Get algorithm data
        all_algs = self.get_all_algorithms()
        alg_dict = {alg['name']: alg for alg in all_algs}
        
        alg1 = alg_dict.get(alg1_name)
        alg2 = alg_dict.get(alg2_name)
        alg3 = alg_dict.get(alg3_name)
        
        if not all([alg1, alg2, alg3]):
            return {"error": "One or more algorithms not found"}
        
        # Analyze noise type coverage
        noise_coverage = set()
        for alg in [alg1, alg2, alg3]:
            if 'all' in alg['noise_types']:
                noise_coverage.update(['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson'])
            else:
                noise_coverage.update(alg['noise_types'])
        
        # Analyze approach diversity
        categories = set([alg['category'] for alg in [alg1, alg2, alg3]])
        
        # Analyze computational balance
        comp_scores = [alg['computational_score'] for alg in [alg1, alg2, alg3]]
        quality_scores = [alg['quality_score'] for alg in [alg1, alg2, alg3]]
        adaptability_scores = [alg['adaptability_score'] for alg in [alg1, alg2, alg3]]
        
        # Calculate compatibility metrics
        noise_coverage_score = len(noise_coverage) / 5.0  # 5 main noise types
        category_diversity_score = len(categories) / 3.0  # Ideally 3 different categories
        computational_balance = 1.0 - np.std(comp_scores)  # Lower std = better balance
        quality_average = np.mean(quality_scores)
        adaptability_average = np.mean(adaptability_scores)
        
        # Overall compatibility score
        compatibility_score = (
            0.25 * noise_coverage_score +
            0.20 * category_diversity_score +
            0.15 * computational_balance +
            0.25 * quality_average +
            0.15 * adaptability_average
        )
        
        return {
            'algorithms': [alg1_name, alg2_name, alg3_name],
            'noise_coverage': list(noise_coverage),
            'noise_coverage_score': noise_coverage_score,
            'category_diversity': list(categories),
            'category_diversity_score': category_diversity_score,
            'computational_scores': comp_scores,
            'computational_balance': computational_balance,
            'quality_scores': quality_scores,
            'quality_average': quality_average,
            'adaptability_scores': adaptability_scores,
            'adaptability_average': adaptability_average,
            'overall_compatibility': compatibility_score,
            'recommendation': self._get_recommendation(compatibility_score)
        }
    
    def _get_recommendation(self, score):
        """Convert compatibility score to recommendation"""
        if score >= 0.8:
            return "Excellent - Highly recommended combination"
        elif score >= 0.7:
            return "Very Good - Strong complementarity"
        elif score >= 0.6:
            return "Good - Reasonable combination"
        elif score >= 0.5:
            return "Fair - Some limitations"
        else:
            return "Poor - Consider alternatives"
    
    def generate_algorithm_comparison_table(self):
        """Generate comprehensive comparison table"""
        all_algs = self.get_all_algorithms()
        
        comparison_data = []
        for alg in all_algs:
            comparison_data.append({
                'Algorithm': alg['name'],
                'Category': alg['category'],
                'Complexity': alg['complexity'],
                'Implementation': alg['implementation_difficulty'],
                'Noise Types': ', '.join(alg['noise_types']),
                'Edge Preservation': alg['edge_preservation'],
                'PSNR Gain': alg['typical_psnr_gain'],
                'Computational Score': alg['computational_score'],
                'Quality Score': alg['quality_score'],
                'Adaptability Score': alg['adaptability_score'],
                'Overall Score': (alg['computational_score'] + alg['quality_score'] + alg['adaptability_score']) / 3.0
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Overall Score', ascending=False)
        df.to_csv('algorithms/algorithm_comparison.csv', index=False)
        
        print("=== ALGORITHM COMPARISON TABLE ===")
        print(df.to_string(index=False))
        print(f"\nTable saved to: algorithms/algorithm_comparison.csv")
        
        return df
    
    def evaluate_all_combinations(self, top_n=10):
        """Evaluate all possible 3-algorithm combinations"""
        all_algs = self.get_all_algorithms()
        alg_names = [alg['name'] for alg in all_algs]
        
        combinations = []
        results = []
        
        # Generate all combinations of 3 algorithms
        for i in range(len(alg_names)):
            for j in range(i+1, len(alg_names)):
                for k in range(j+1, len(alg_names)):
                    combo = (alg_names[i], alg_names[j], alg_names[k])
                    combinations.append(combo)
                    
                    # Analyze this combination
                    analysis = self.analyze_compatibility(combo[0], combo[1], combo[2])
                    if 'error' not in analysis:
                        results.append({
                            'combination': combo,
                            'score': analysis['overall_compatibility'],
                            'analysis': analysis
                        })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n=== TOP {top_n} ALGORITHM COMBINATIONS ===")
        for i, result in enumerate(results[:top_n]):
            combo = result['combination']
            score = result['score']
            analysis = result['analysis']
            
            print(f"\n{i+1}. {combo[0]} + {combo[1]} + {combo[2]}")
            print(f"   Score: {score:.3f} - {analysis['recommendation']}")
            print(f"   Noise Coverage: {analysis['noise_coverage']}")
            print(f"   Categories: {analysis['category_diversity']}")
            print(f"   Quality Avg: {analysis['quality_average']:.2f}")
        
        return results[:top_n]

def main():
    """Run comprehensive algorithm analysis"""
    print("🔬 ALGORITHM ANALYSIS FRAMEWORK")
    print("=" * 50)
    
    analyzer = AlgorithmAnalyzer()
    
    # Generate comparison table
    print("\n1. Generating algorithm comparison table...")
    comparison_df = analyzer.generate_algorithm_comparison_table()
    
    # Find top combinations
    print("\n2. Finding optimal algorithm combinations...")
    top_combinations = analyzer.evaluate_all_combinations(top_n=5)
    
    # Analyze specific promising combinations
    print("\n3. Detailed analysis of top candidates...")
    
    promising_combinations = [
        ('adaptive_bilateral', 'multi_method_consensus', 'edge_preserving_nlm'),
        ('bilateral_filter', 'non_local_means', 'lee_filter'),
        ('median_filter', 'bilateral_filter', 'bm3d_simplified'),
        ('adaptive_bilateral', 'weighted_median_consensus', 'edge_preserving_nlm')
    ]
    
    detailed_analyses = []
    for combo in promising_combinations:
        analysis = analyzer.analyze_compatibility(combo[0], combo[1], combo[2])
        detailed_analyses.append(analysis)
        
        print(f"\n--- {combo[0]} + {combo[1]} + {combo[2]} ---")
        print(f"Score: {analysis['overall_compatibility']:.3f}")
        print(f"Recommendation: {analysis['recommendation']}")
        print(f"Noise Coverage: {analysis['noise_coverage']}")
    
    # Save detailed results
    with open('algorithms/detailed_analysis.json', 'w') as f:
        json.dump({
            'top_combinations': [
                {
                    'combination': result['combination'],
                    'score': result['score'],
                    'analysis': result['analysis']
                }
                for result in top_combinations
            ],
            'detailed_analyses': detailed_analyses,
            'analysis_timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Analysis complete! Results saved to algorithms/detailed_analysis.json")
    print(f"📊 Comparison table: algorithms/algorithm_comparison.csv")

if __name__ == "__main__":
    main()
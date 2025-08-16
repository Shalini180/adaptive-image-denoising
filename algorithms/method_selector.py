import pandas as pd
import numpy as np
import json
from algorithm_analyzer import AlgorithmAnalyzer

class MethodSelector:
    """
    Advanced method selection framework for adaptive denoising system
    Uses multi-criteria decision analysis to select optimal 3-method combination
    """
    
    def __init__(self):
        self.selection_criteria = {
            'complementarity': {
                'weight': 0.25,
                'description': 'How well methods complement each other',
                'metrics': ['noise_type_coverage', 'approach_diversity', 'strength_combination'],
                'target_score': 0.9  # Want near-complete coverage
            },
            'implementation_feasibility': {
                'weight': 0.20,
                'description': 'Ease of implementation and integration',
                'metrics': ['complexity_balance', 'parameter_sensitivity', 'debugging_difficulty'],
                'target_score': 0.7  # Balance quality vs. complexity
            },
            'performance_potential': {
                'weight': 0.25,
                'description': 'Expected denoising quality',
                'metrics': ['quality_scores', 'edge_preservation', 'psnr_potential'],
                'target_score': 0.8  # High quality requirement
            },
            'adaptability': {
                'weight': 0.20,
                'description': 'Ability to adapt to different conditions',
                'metrics': ['parameter_tunability', 'noise_flexibility', 'robustness'],
                'target_score': 0.8  # Must be adaptive
            },
            'computational_efficiency': {
                'weight': 0.10,
                'description': 'Speed and resource requirements',
                'metrics': ['time_complexity', 'memory_usage', 'parallelizability'],
                'target_score': 0.6  # Reasonable performance
            }
        }
        
        self.analyzer = AlgorithmAnalyzer()
        
        # Define strategic method roles
        self.method_roles = {
            'method_a': {
                'role': 'noise_specific_specialist',
                'description': 'Adapts parameters based on detected noise type',
                'requirements': ['high_adaptability', 'multiple_noise_types', 'fast_execution'],
                'target_algorithms': ['adaptive_bilateral', 'bilateral_filter']
            },
            'method_b': {
                'role': 'consensus_coordinator',
                'description': 'Combines multiple approaches for robustness',
                'requirements': ['multi_method_integration', 'robust_averaging', 'all_noise_types'],
                'target_algorithms': ['multi_method_consensus', 'weighted_median_consensus']
            },
            'method_c': {
                'role': 'quality_enhancer',
                'description': 'Provides high-quality refinement for critical regions',
                'requirements': ['excellent_quality', 'edge_preservation', 'texture_preservation'],
                'target_algorithms': ['edge_preserving_nlm', 'non_local_means', 'bm3d_simplified']
            }
        }
    
    def evaluate_method_for_role(self, algorithm_name, role_requirements):
        """Evaluate how well an algorithm fits a specific role"""
        all_algs = self.analyzer.get_all_algorithms()
        alg_dict = {alg['name']: alg for alg in all_algs}
        
        if algorithm_name not in alg_dict:
            return 0.0
        
        alg = alg_dict[algorithm_name]
        score = 0.0
        
        # Check role-specific requirements
        for requirement in role_requirements:
            if requirement == 'high_adaptability':
                score += alg['adaptability_score'] * 0.3
            elif requirement == 'multiple_noise_types':
                noise_count = len(alg['noise_types'])
                if 'all' in alg['noise_types']:
                    noise_count = 5
                score += min(noise_count / 5.0, 1.0) * 0.3
            elif requirement == 'fast_execution':
                score += alg['computational_score'] * 0.2
            elif requirement == 'multi_method_integration':
                if 'consensus' in algorithm_name:
                    score += 0.4
            elif requirement == 'robust_averaging':
                if alg['edge_preservation'] in ['Good', 'Excellent']:
                    score += 0.3
            elif requirement == 'all_noise_types':
                if 'all' in alg['noise_types']:
                    score += 0.3
            elif requirement == 'excellent_quality':
                score += alg['quality_score'] * 0.4
            elif requirement == 'edge_preservation':
                if alg['edge_preservation'] == 'Excellent':
                    score += 0.3
                elif alg['edge_preservation'] == 'Good':
                    score += 0.2
            elif requirement == 'texture_preservation':
                if 'texture' in ' '.join(alg['strengths']).lower():
                    score += 0.3
        
        return min(score, 1.0)
    
    def find_optimal_role_assignments(self):
        """Find the best algorithm for each role"""
        role_assignments = {}
        
        for role_name, role_data in self.method_roles.items():
            best_score = 0.0
            best_algorithm = None
            
            print(f"\n🎯 Evaluating algorithms for {role_name} ({role_data['role']}):")
            print(f"   Requirements: {role_data['requirements']}")
            
            # Evaluate each target algorithm for this role
            for alg_name in role_data['target_algorithms']:
                score = self.evaluate_method_for_role(alg_name, role_data['requirements'])
                print(f"   {alg_name}: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_algorithm = alg_name
            
            role_assignments[role_name] = {
                'algorithm': best_algorithm,
                'score': best_score,
                'role': role_data['role']
            }
            
            print(f"   ✅ Selected: {best_algorithm} (score: {best_score:.3f})")
        
        return role_assignments
    
    def evaluate_combination_detailed(self, method_a, method_b, method_c):
        """Comprehensive evaluation of a 3-method combination"""
        
        # Get compatibility analysis from algorithm analyzer
        compatibility = self.analyzer.analyze_compatibility(method_a, method_b, method_c)
        if 'error' in compatibility:
            return None
        
        scores = {}
        
        # 1. Complementarity Score (25%)
        noise_coverage_score = compatibility['noise_coverage_score']
        category_diversity_score = compatibility['category_diversity_score']
        complementarity_score = (noise_coverage_score + category_diversity_score) / 2.0
        scores['complementarity'] = complementarity_score
        
        # 2. Implementation Feasibility (20%)
        all_algs = self.analyzer.get_all_algorithms()
        alg_dict = {alg['name']: alg for alg in all_algs}
        
        # Convert implementation difficulty to numeric scores
        difficulty_scores = {
            'Easy': 1.0, 'Medium': 0.7, 'Hard': 0.4, 'Very Hard': 0.1
        }
        
        impl_scores = []
        for alg_name in [method_a, method_b, method_c]:
            if alg_name in alg_dict:
                difficulty = alg_dict[alg_name]['implementation_difficulty']
                impl_scores.append(difficulty_scores.get(difficulty, 0.5))
        
        implementation_score = np.mean(impl_scores) if impl_scores else 0.0
        scores['implementation_feasibility'] = implementation_score
        
        # 3. Performance Potential (25%)
        quality_scores = []
        for alg_name in [method_a, method_b, method_c]:
            if alg_name in alg_dict:
                quality_scores.append(alg_dict[alg_name]['quality_score'])
        
        performance_score = np.mean(quality_scores) if quality_scores else 0.0
        scores['performance_potential'] = performance_score
        
        # 4. Adaptability (20%)
        adaptability_scores = []
        for alg_name in [method_a, method_b, method_c]:
            if alg_name in alg_dict:
                adaptability_scores.append(alg_dict[alg_name]['adaptability_score'])
        
        adaptability_score = np.mean(adaptability_scores) if adaptability_scores else 0.0
        scores['adaptability'] = adaptability_score
        
        # 5. Computational Efficiency (10%)
        computational_scores = []
        for alg_name in [method_a, method_b, method_c]:
            if alg_name in alg_dict:
                computational_scores.append(alg_dict[alg_name]['computational_score'])
        
        efficiency_score = np.mean(computational_scores) if computational_scores else 0.0
        scores['computational_efficiency'] = efficiency_score
        
        # Calculate weighted final score
        final_score = sum(
            scores[criterion] * criteria_data['weight']
            for criterion, criteria_data in self.selection_criteria.items()
        )
        
        # Gap analysis
        gaps = {}
        for criterion, criteria_data in self.selection_criteria.items():
            gap = criteria_data['target_score'] - scores[criterion]
            gaps[criterion] = max(gap, 0.0)  # Only positive gaps
        
        return {
            'combination': [method_a, method_b, method_c],
            'final_score': final_score,
            'detailed_scores': scores,
            'gaps': gaps,
            'recommendation': self._get_detailed_recommendation(final_score, gaps),
            'compatibility_data': compatibility
        }
    
    def _get_detailed_recommendation(self, score, gaps):
        """Generate detailed recommendation based on score and gaps"""
        if score >= 0.8 and max(gaps.values()) < 0.1:
            return {
                'level': 'Excellent',
                'description': 'Highly recommended - meets all criteria',
                'confidence': 'High'
            }
        elif score >= 0.7 and max(gaps.values()) < 0.2:
            return {
                'level': 'Very Good',
                'description': 'Strong candidate with minor gaps',
                'confidence': 'High'
            }
        elif score >= 0.6:
            return {
                'level': 'Good',
                'description': 'Viable option with some limitations',
                'confidence': 'Medium'
            }
        elif score >= 0.5:
            return {
                'level': 'Fair',
                'description': 'Acceptable but consider improvements',
                'confidence': 'Low'
            }
        else:
            return {
                'level': 'Poor',
                'description': 'Significant limitations - seek alternatives',
                'confidence': 'Very Low'
            }
    
    def select_final_methods(self):
        """Execute complete method selection process"""
        print("🔍 METHOD SELECTION FRAMEWORK")
        print("=" * 50)
        
        # Step 1: Role-based assignments
        print("\n1. ROLE-BASED ALGORITHM ASSIGNMENT")
        role_assignments = self.find_optimal_role_assignments()
        
        # Step 2: Extract recommended combination
        recommended_combination = (
            role_assignments['method_a']['algorithm'],
            role_assignments['method_b']['algorithm'],
            role_assignments['method_c']['algorithm']
        )
        
        print(f"\n📋 RECOMMENDED COMBINATION:")
        print(f"   Method A (Noise Specialist): {recommended_combination[0]}")
        print(f"   Method B (Consensus): {recommended_combination[1]}")
        print(f"   Method C (Quality Enhancer): {recommended_combination[2]}")
        
        # Step 3: Detailed evaluation of recommended combination
        print(f"\n2. DETAILED EVALUATION")
        evaluation = self.evaluate_combination_detailed(*recommended_combination)
        
        if evaluation:
            print(f"   Final Score: {evaluation['final_score']:.3f}")
            print(f"   Recommendation: {evaluation['recommendation']['level']}")
            print(f"   Confidence: {evaluation['recommendation']['confidence']}")
            
            print(f"\n   Detailed Scores:")
            for criterion, score in evaluation['detailed_scores'].items():
                weight = self.selection_criteria[criterion]['weight']
                target = self.selection_criteria[criterion]['target_score']
                gap = evaluation['gaps'][criterion]
                print(f"     {criterion}: {score:.3f} (weight: {weight:.1%}, target: {target:.2f}, gap: {gap:.3f})")
            
            # Step 4: Alternative combinations analysis
            print(f"\n3. ALTERNATIVE COMBINATIONS ANALYSIS")
            
            alternative_combinations = [
                ('bilateral_filter', 'multi_method_consensus', 'non_local_means'),
                ('adaptive_bilateral', 'weighted_median_consensus', 'bm3d_simplified'),
                ('median_filter', 'multi_method_consensus', 'edge_preserving_nlm')
            ]
            
            alternative_evaluations = []
            for combo in alternative_combinations:
                alt_eval = self.evaluate_combination_detailed(*combo)
                if alt_eval:
                    alternative_evaluations.append(alt_eval)
                    print(f"   {combo}: {alt_eval['final_score']:.3f} ({alt_eval['recommendation']['level']})")
            
            # Step 5: Final selection and justification
            print(f"\n4. FINAL SELECTION & JUSTIFICATION")
            
            # Compare with alternatives
            all_evaluations = [evaluation] + alternative_evaluations
            best_evaluation = max(all_evaluations, key=lambda x: x['final_score'])
            
            final_selection = {
                'selected_methods': best_evaluation['combination'],
                'selection_score': best_evaluation['final_score'],
                'selection_rationale': self._generate_selection_rationale(best_evaluation),
                'role_assignments': role_assignments,
                'evaluation_details': best_evaluation,
                'alternatives_considered': alternative_evaluations
            }
            
            print(f"   🎯 FINAL SELECTION: {final_selection['selected_methods']}")
            print(f"   📊 Selection Score: {final_selection['selection_score']:.3f}")
            print(f"\n   🔍 RATIONALE:")
            for point in final_selection['selection_rationale']:
                print(f"     • {point}")
            
            return final_selection
        
        return None
    
    def _generate_selection_rationale(self, evaluation):
        """Generate human-readable rationale for selection"""
        rationale = []
        
        scores = evaluation['detailed_scores']
        gaps = evaluation['gaps']
        
        # Complementarity
        if scores['complementarity'] >= 0.8:
            rationale.append(f"Excellent complementarity ({scores['complementarity']:.2f}) - covers diverse noise types and approaches")
        elif gaps['complementarity'] > 0.1:
            rationale.append(f"Complementarity needs improvement (gap: {gaps['complementarity']:.2f}) - consider broader noise coverage")
        
        # Implementation
        if scores['implementation_feasibility'] >= 0.7:
            rationale.append(f"Good implementation feasibility ({scores['implementation_feasibility']:.2f}) - balanced complexity")
        elif gaps['implementation_feasibility'] > 0.1:
            rationale.append(f"Implementation may be challenging (gap: {gaps['implementation_feasibility']:.2f}) - consider simpler alternatives")
        
        # Performance
        if scores['performance_potential'] >= 0.8:
            rationale.append(f"High performance potential ({scores['performance_potential']:.2f}) - strong quality scores")
        
        # Adaptability
        if scores['adaptability'] >= 0.8:
            rationale.append(f"Excellent adaptability ({scores['adaptability']:.2f}) - handles diverse conditions well")
        
        # Overall assessment
        if evaluation['final_score'] >= 0.75:
            rationale.append("Overall: Strong combination meeting research objectives")
        elif evaluation['final_score'] >= 0.65:
            rationale.append("Overall: Good combination with acceptable trade-offs")
        else:
            rationale.append("Overall: Consider refinements to improve performance")
        
        return rationale

def main():
    """Execute method selection process"""
    selector = MethodSelector()
    
    # Run complete selection process
    final_selection = selector.select_final_methods()
    
    if final_selection:
        # Save results
        with open('algorithms/final_method_selection.json', 'w') as f:
            json.dump(final_selection, f, indent=2, default=str)
        
        print(f"\n✅ Method selection complete!")
        print(f"📁 Results saved to: algorithms/final_method_selection.json")
        
        return final_selection
    else:
        print("❌ Method selection failed!")
        return None

if __name__ == "__main__":
    main()
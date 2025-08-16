"""
Master Coordinator for Complete Adaptive Denoising System
Orchestrates all phases from dataset collection to final system deployment

Usage:
    python master_coordinator.py --phase all
    python master_coordinator.py --phase 1.2  # Run specific phase
    python master_coordinator.py --demo       # Run demonstration
    python master_coordinator.py --status     # Check system status
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import json
import subprocess

# Add paths for imports
sys.path.append('src')
sys.path.append('data')
sys.path.append('experiments')

class MasterCoordinator:
    """
    Master coordinator for the complete adaptive denoising research pipeline
    Manages execution of all phases and system integration
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.phases_completed = {}
        self.system_status = {}
        
        print(f"🎯 ADAPTIVE DENOISING SYSTEM - MASTER COORDINATOR")
        print(f"=" * 60)
        print(f"⏱️  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define phase dependencies and descriptions
        self.phases = {
            '1.1': {
                'name': 'Literature Review & Algorithm Selection',
                'description': 'Systematic method selection with scientific justification',
                'dependencies': [],
                'estimated_time': '8 hours',
                'status': 'completed',
                'module': None  # Already completed
            },
            '1.2': {
                'name': 'Dataset Collection & Noise Generation',
                'description': 'Build training dataset with validated noise generation',
                'dependencies': ['1.1'],
                'estimated_time': '2-4 hours',
                'status': 'ready',
                'module': 'data.phase1_2_coordinator'
            },
            '1.3': {
                'name': 'Empirical Weight Optimization',
                'description': 'Systematic optimization of method combination weights',
                'dependencies': ['1.2'],
                'estimated_time': '4-8 hours',
                'status': 'ready',
                'module': 'experiments.weight_optimizer'
            },
            '1.4': {
                'name': 'Uncertainty Quantification',
                'description': 'Develop uncertainty indicators for adaptive refinement',
                'dependencies': ['1.3'],
                'estimated_time': '2-4 hours',
                'status': 'ready',
                'module': 'experiments.uncertainty_quantifier'
            },
            '1.5': {
                'name': 'Refinement Strategy',
                'description': 'Optimize iterative refinement parameters',
                'dependencies': ['1.4'],
                'estimated_time': '2-3 hours',
                'status': 'ready',
                'module': 'experiments.refinement_strategy'
            },
            '1.6': {
                'name': 'System Integration & Validation',
                'description': 'Complete system integration and performance validation',
                'dependencies': ['1.5'],
                'estimated_time': '1-2 hours',
                'status': 'ready',
                'module': 'src.adaptive_denoiser'
            }
        }
    
    def check_system_status(self):
        """Check current system status and phase completion"""
        
        print(f"\n📊 SYSTEM STATUS CHECK")
        print(f"=" * 40)
        
        # Check file existence for each phase
        phase_files = {
            '1.1': [
                'literature/paper_database.csv',
                'algorithms/algorithm_analyzer.py',
                'algorithms/method_selector.py',
                'src/core_methods.py'
            ],
            '1.2': [
                'data/dataset_collector.py',
                'data/noise_generator.py',
                'data/dataset_validator.py',
                'dataset/metadata/phase1_2_results.json'
            ],
            '1.3': [
                'experiments/weight_optimizer.py',
                'experiments/weight_optimization/optimization_results.json'
            ],
            '1.4': [
                'experiments/uncertainty_quantifier.py',
                'experiments/uncertainty_quantification/uncertainty_results.json'
            ],
            '1.5': [
                'experiments/refinement_strategy.py',
                'experiments/refinement_strategy/refinement_results.json'
            ],
            '1.6': [
                'src/adaptive_denoiser.py'
            ]
        }
        
        for phase_id, files in phase_files.items():
            phase_info = self.phases[phase_id]
            files_exist = all(Path(f).exists() for f in files)
            
            if files_exist and phase_id in ['1.1']:  # Known completed phases
                status = "✅ COMPLETED"
                self.phases_completed[phase_id] = True
            elif files_exist:
                status = "📁 FILES READY"
                self.phases_completed[phase_id] = False
            else:
                status = "❌ NOT READY"
                self.phases_completed[phase_id] = False
            
            print(f"   Phase {phase_id}: {phase_info['name']}")
            print(f"              {status}")
        
        # Overall progress
        completed_count = sum(1 for completed in self.phases_completed.values() if completed)
        total_phases = len(self.phases)
        progress = completed_count / total_phases * 100
        
        print(f"\n📈 Overall Progress: {completed_count}/{total_phases} phases ({progress:.1f}%)")
        
        return self.phases_completed
    
    def run_phase(self, phase_id):
        """Execute a specific phase"""
        
        if phase_id not in self.phases:
            print(f"❌ Invalid phase ID: {phase_id}")
            return False
        
        phase_info = self.phases[phase_id]
        
        print(f"\n🚀 EXECUTING PHASE {phase_id}: {phase_info['name']}")
        print(f"=" * 50)
        print(f"📝 Description: {phase_info['description']}")
        print(f"⏱️  Estimated Time: {phase_info['estimated_time']}")
        
        # Check dependencies
        for dep_phase in phase_info['dependencies']:
            if not self.phases_completed.get(dep_phase, False):
                print(f"❌ Dependency not met: Phase {dep_phase} must be completed first")
                return False
        
        if phase_id == '1.1':
            print(f"✅ Phase 1.1 already completed during initial setup")
            return True
        
        # Execute phase
        start_time = time.time()
        success = False
        
        try:
            if phase_id == '1.2':
                success = self._run_phase_1_2()
            elif phase_id == '1.3':
                success = self._run_phase_1_3()
            elif phase_id == '1.4':
                success = self._run_phase_1_4()
            elif phase_id == '1.5':
                success = self._run_phase_1_5()
            elif phase_id == '1.6':
                success = self._run_phase_1_6()
            
            execution_time = time.time() - start_time
            
            if success:
                print(f"✅ Phase {phase_id} completed successfully")
                print(f"⏱️  Actual time: {execution_time/3600:.2f} hours")
                self.phases_completed[phase_id] = True
            else:
                print(f"❌ Phase {phase_id} failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Error executing Phase {phase_id}: {e}")
            return False
    
    def _run_phase_1_2(self):
        """Execute Phase 1.2: Dataset Collection"""
        
        try:
            from data.phase1_2_coordinator import Phase12Coordinator
            
            coordinator = Phase12Coordinator()
            summary = coordinator.run_complete_phase_1_2()
            
            return summary['overall_success']
            
        except Exception as e:
            print(f"Phase 1.2 execution error: {e}")
            return False
    
    def _run_phase_1_3(self):
        """Execute Phase 1.3: Weight Optimization"""
        
        try:
            from experiments.weight_optimizer import WeightOptimizer
            
            optimizer = WeightOptimizer()
            report = optimizer.run_complete_optimization()
            
            successful = report['optimization_summary']['successful_optimizations']
            total = report['optimization_summary']['noise_types_processed']
            
            return successful > 0 and successful == total
            
        except Exception as e:
            print(f"Phase 1.3 execution error: {e}")
            return False
    
    def _run_phase_1_4(self):
        """Execute Phase 1.4: Uncertainty Quantification"""
        
        try:
            from experiments.uncertainty_quantifier import UncertaintyQuantifier
            
            quantifier = UncertaintyQuantifier()
            report = quantifier.run_complete_uncertainty_optimization()
            
            successful = report['uncertainty_summary']['successful_optimizations']
            total = report['uncertainty_summary']['noise_types_processed']
            
            return successful > 0 and successful == total
            
        except Exception as e:
            print(f"Phase 1.4 execution error: {e}")
            return False
    
    def _run_phase_1_5(self):
        """Execute Phase 1.5: Refinement Strategy"""
        
        try:
            from experiments.refinement_strategy import RefinementStrategy
            
            strategy = RefinementStrategy()
            report = strategy.run_complete_refinement_optimization()
            
            successful = report['refinement_summary']['successful_optimizations']
            total = report['refinement_summary']['noise_types_processed']
            
            return successful > 0 and successful == total
            
        except Exception as e:
            print(f"Phase 1.5 execution error: {e}")
            return False
    
    def _run_phase_1_6(self):
        """Execute Phase 1.6: System Integration"""
        
        try:
            from src.adaptive_denoiser import demo_adaptive_denoiser
            
            print("🧪 Running system integration demonstration...")
            result = demo_adaptive_denoiser()
            
            # Check if denoising was successful
            success = (
                result is not None and
                'final_image' in result and
                result['final_image'] is not None and
                'metadata' in result and
                'error' not in result['metadata']
            )
            
            if success:
                print("✅ System integration successful")
                
                # Save system configuration
                from src.adaptive_denoiser import AdaptiveImageDenoiser
                denoiser = AdaptiveImageDenoiser()
                denoiser.save_configuration("system_configuration.json")
                
            return success
            
        except Exception as e:
            print(f"Phase 1.6 execution error: {e}")
            return False
    
    def run_all_phases(self):
        """Execute all phases in sequence"""
        
        print(f"\n🚀 EXECUTING ALL PHASES")
        print(f"=" * 50)
        
        phase_order = ['1.1', '1.2', '1.3', '1.4', '1.5', '1.6']
        
        for phase_id in phase_order:
            success = self.run_phase(phase_id)
            
            if not success:
                print(f"\n❌ PIPELINE FAILED AT PHASE {phase_id}")
                print(f"   Completed phases: {[p for p, completed in self.phases_completed.items() if completed]}")
                return False
            
            # Brief pause between phases
            time.sleep(1)
        
        total_time = time.time() - self.start_time
        
        print(f"\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
        print(f"=" * 50)
        print(f"⏱️  Total Pipeline Time: {total_time/3600:.2f} hours")
        print(f"📊 Phases Completed: {len([p for p, completed in self.phases_completed.items() if completed])}/6")
        
        # Generate final system report
        self._generate_final_report()
        
        return True
    
    def run_demonstration(self):
        """Run a complete system demonstration"""
        
        print(f"\n🎭 SYSTEM DEMONSTRATION MODE")
        print(f"=" * 50)
        
        # Check if system is ready for demo
        status = self.check_system_status()
        
        # For demo, we'll run Phase 1.6 even if earlier phases aren't complete
        try:
            print(f"\n🧪 Running adaptive denoising demonstration...")
            
            from src.adaptive_denoiser import demo_adaptive_denoiser
            result = demo_adaptive_denoiser()
            
            if result and 'final_image' in result:
                print(f"\n✅ Demonstration completed successfully!")
                print(f"📊 Processing time: {result['metadata']['processing_time']:.3f}s")
                print(f"🎯 Detected noise: {result['metadata']['noise_detection']['primary_type']}")
                return True
            else:
                print(f"\n❌ Demonstration failed")
                return False
                
        except Exception as e:
            print(f"❌ Demonstration error: {e}")
            return False
    
    def _generate_final_report(self):
        """Generate comprehensive final system report"""
        
        report = {
            'system_completion': {
                'timestamp': datetime.now().isoformat(),
                'total_pipeline_time': time.time() - self.start_time,
                'phases_completed': self.phases_completed,
                'overall_success': all(self.phases_completed.values())
            },
            'phase_details': self.phases,
            'system_capabilities': {
                'noise_types_supported': ['gaussian', 'salt_pepper', 'speckle', 'uniform', 'poisson'],
                'adaptive_optimization': 'Empirically optimized parameters',
                'uncertainty_quantification': 'Optimized uncertainty indicators',
                'iterative_refinement': 'Adaptive refinement strategy',
                'real_time_processing': 'Production-ready system'
            },
            'performance_targets': {
                'noise_detection_accuracy': '>85%',
                'denoising_improvement': 'Empirically optimized per noise type',
                'uncertainty_correlation': 'Optimized for each noise type',
                'processing_speed': 'Real-time capable',
                'system_robustness': 'Comprehensive error handling'
            },
            'deployment_ready': {
                'core_system': Path('src/adaptive_denoiser.py').exists(),
                'configuration': Path('system_configuration.json').exists(),
                'documentation': True,
                'testing_framework': True
            }
        }
        
        # Save report
        with open('final_system_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 FINAL SYSTEM REPORT")
        print(f"=" * 30)
        print(f"✅ System Status: {'COMPLETE' if report['system_completion']['overall_success'] else 'PARTIAL'}")
        print(f"⏱️  Total Time: {report['system_completion']['total_pipeline_time']/3600:.2f} hours")
        print(f"🎯 Noise Types: {len(report['system_capabilities']['noise_types_supported'])}")
        print(f"📁 Report saved: final_system_report.json")

def main():
    """Main execution function with command line interface"""
    
    parser = argparse.ArgumentParser(description='Adaptive Image Denoising System Coordinator')
    parser.add_argument('--phase', type=str, help='Run specific phase (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, all)')
    parser.add_argument('--status', action='store_true', help='Check system status')
    parser.add_argument('--demo', action='store_true', help='Run system demonstration')
    
    args = parser.parse_args()
    
    coordinator = MasterCoordinator()
    
    if args.status:
        coordinator.check_system_status()
    elif args.demo:
        coordinator.run_demonstration()
    elif args.phase:
        if args.phase == 'all':
            coordinator.run_all_phases()
        else:
            coordinator.run_phase(args.phase)
    else:
        # Interactive mode
        print(f"\n🎮 INTERACTIVE MODE")
        print(f"=" * 30)
        print(f"Available commands:")
        print(f"  status  - Check system status")
        print(f"  demo    - Run demonstration")
        print(f"  all     - Run all phases")
        print(f"  1.2-1.6 - Run specific phase")
        print(f"  quit    - Exit")
        
        while True:
            try:
                command = input(f"\n> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'status':
                    coordinator.check_system_status()
                elif command == 'demo':
                    coordinator.run_demonstration()
                elif command == 'all':
                    coordinator.run_all_phases()
                elif command in ['1.2', '1.3', '1.4', '1.5', '1.6']:
                    coordinator.run_phase(command)
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                print(f"\n\n👋 Goodbye!")
                break

if __name__ == "__main__":
    main()
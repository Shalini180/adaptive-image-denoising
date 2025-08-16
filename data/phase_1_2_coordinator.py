"""
Phase 1.2 Coordinator: Complete Dataset Collection Process
Orchestrates the entire dataset collection, noise generation, and validation workflow

Target: 10,000+ clean images → 300,000+ noisy training pairs
Quality: >95% noise generation accuracy with comprehensive validation
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Import our dataset modules
from dataset_collector import DatasetCollector
from noise_generator import NoiseGenerator
from dataset_validator import DatasetValidator

class Phase12Coordinator:
    """
    Orchestrates the complete Phase 1.2 dataset collection process
    Manages workflow from clean image collection to final validation
    """
    
    def __init__(self, dataset_dir="dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.start_time = time.time()
        
        # Initialize components
        self.collector = DatasetCollector(dataset_dir)
        self.generator = NoiseGenerator(dataset_dir)
        self.validator = DatasetValidator(dataset_dir)
        
        # Phase tracking
        self.phase_results = {
            'start_time': datetime.now().isoformat(),
            'collection': None,
            'generation': None,
            'validation': None,
            'overall_success': False
        }
        
        print(f"🎯 PHASE 1.2 COORDINATOR")
        print(f"=" * 50)
        print(f"📁 Dataset Directory: {self.dataset_dir}")
        print(f"⏱️  Process Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def execute_collection_phase(self):
        """Execute clean image collection"""
        print(f"\n🗂️  PHASE 1.2.1: CLEAN IMAGE COLLECTION")
        print(f"=" * 40)
        
        collection_start = time.time()
        
        try:
            # Run collection process
            total_collected = self.collector.collect_clean_images()
            
            # Get collection status
            status = self.collector.get_collection_status()
            
            collection_time = time.time() - collection_start
            
            # Determine success criteria
            collection_success = (
                total_collected > 0 and 
                status['total_clean'] >= 50  # Minimum viable dataset
            )
            
            self.phase_results['collection'] = {
                'success': collection_success,
                'total_collected': total_collected,
                'final_status': status,
                'processing_time': collection_time,
                'completion_rate': status['completion_percentage'] / 100.0
            }
            
            print(f"\n📊 COLLECTION PHASE SUMMARY:")
            print(f"   Images Collected: {total_collected}")
            print(f"   Total Clean Images: {status['total_clean']}")
            print(f"   Target Progress: {status['completion_percentage']:.1f}%")
            print(f"   Processing Time: {collection_time:.1f}s")
            print(f"   Status: {'SUCCESS' if collection_success else 'PARTIAL'}")
            
            return collection_success
            
        except Exception as e:
            print(f"   ❌ Collection phase failed: {e}")
            self.phase_results['collection'] = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - collection_start
            }
            return False
    
    def execute_generation_phase(self):
        """Execute noise generation process"""
        print(f"\n🎲 PHASE 1.2.2: NOISE GENERATION")
        print(f"=" * 40)
        
        generation_start = time.time()
        
        try:
            # Run noise generation
            results = self.generator.process_all_clean_images()
            
            generation_time = time.time() - generation_start
            
            # Determine success criteria
            generation_success = (
                results and 
                results['total_pairs_generated'] > 0 and
                results['success_rate'] >= 0.8  # 80% success rate minimum
            )
            
            self.phase_results['generation'] = {
                'success': generation_success,
                'results': results,
                'processing_time': generation_time
            }
            
            if results:
                print(f"\n📊 GENERATION PHASE SUMMARY:")
                print(f"   Training Pairs Generated: {results['total_pairs_generated']:,}")
                print(f"   Success Rate: {results['success_rate']*100:.1f}%")
                print(f"   Successful Images: {results['successful_images']}")
                print(f"   Processing Time: {generation_time:.1f}s")
                print(f"   Status: {'SUCCESS' if generation_success else 'PARTIAL'}")
            
            return generation_success
            
        except Exception as e:
            print(f"   ❌ Generation phase failed: {e}")
            self.phase_results['generation'] = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - generation_start
            }
            return False
    
    def execute_validation_phase(self):
        """Execute dataset validation"""
        print(f"\n🔍 PHASE 1.2.3: DATASET VALIDATION")
        print(f"=" * 40)
        
        validation_start = time.time()
        
        try:
            # Run comprehensive validation
            report = self.validator.run_complete_validation()
            
            validation_time = time.time() - validation_start
            validation_success = report['validation_passed']
            
            self.phase_results['validation'] = {
                'success': validation_success,
                'report': report,
                'processing_time': validation_time
            }
            
            print(f"\n📊 VALIDATION PHASE SUMMARY:")
            print(f"   Overall Score: {report['overall_validation_score']:.3f}")
            print(f"   Quality Threshold: 0.800")
            print(f"   Processing Time: {validation_time:.1f}s")
            print(f"   Status: {'PASSED' if validation_success else 'FAILED'}")
            
            return validation_success
            
        except Exception as e:
            print(f"   ❌ Validation phase failed: {e}")
            self.phase_results['validation'] = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - validation_start
            }
            return False
    
    def generate_phase_summary(self):
        """Generate comprehensive Phase 1.2 summary"""
        total_time = time.time() - self.start_time
        
        # Determine overall success
        all_phases_success = (
            self.phase_results['collection'] and self.phase_results['collection']['success'] and
            self.phase_results['generation'] and self.phase_results['generation']['success'] and
            self.phase_results['validation'] and self.phase_results['validation']['success']
        )
        
        self.phase_results['overall_success'] = all_phases_success
        self.phase_results['total_processing_time'] = total_time
        self.phase_results['end_time'] = datetime.now().isoformat()
        
        # Extract key metrics
        summary_metrics = {
            'total_clean_images': 0,
            'total_noisy_pairs': 0,
            'noise_accuracy': 0.0,
            'validation_score': 0.0,
            'processing_time_hours': total_time / 3600.0
        }
        
        if self.phase_results['collection'] and 'final_status' in self.phase_results['collection']:
            summary_metrics['total_clean_images'] = self.phase_results['collection']['final_status']['total_clean']
        
        if self.phase_results['generation'] and 'results' in self.phase_results['generation']:
            gen_results = self.phase_results['generation']['results']
            if gen_results:
                summary_metrics['total_noisy_pairs'] = gen_results['total_pairs_generated']
        
        if self.phase_results['validation'] and 'report' in self.phase_results['validation']:
            val_report = self.phase_results['validation']['report']
            summary_metrics['validation_score'] = val_report['overall_validation_score']
            
            # Extract noise accuracy if available
            if 'detailed_results' in val_report and 'noise_accuracy' in val_report['detailed_results']:
                summary_metrics['noise_accuracy'] = val_report['detailed_results']['noise_accuracy']['overall_accuracy']
        
        # Save complete results
        results_path = self.dataset_dir / "metadata" / "phase1_2_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.phase_results, f, indent=2, default=str)
        
        # Print comprehensive summary
        print(f"\n🎯 PHASE 1.2 COMPLETE SUMMARY")
        print(f"=" * 50)
        print(f"⏱️  Total Processing Time: {total_time/3600.0:.2f} hours")
        print(f"📊 Key Metrics:")
        print(f"   Clean Images: {summary_metrics['total_clean_images']:,}")
        print(f"   Noisy Training Pairs: {summary_metrics['total_noisy_pairs']:,}")
        print(f"   Noise Generation Accuracy: {summary_metrics['noise_accuracy']*100:.1f}%")
        print(f"   Validation Score: {summary_metrics['validation_score']:.3f}")
        
        print(f"\n📋 Phase Results:")
        print(f"   Collection: {'✅ SUCCESS' if self.phase_results['collection'] and self.phase_results['collection']['success'] else '❌ FAILED'}")
        print(f"   Generation: {'✅ SUCCESS' if self.phase_results['generation'] and self.phase_results['generation']['success'] else '❌ FAILED'}")
        print(f"   Validation: {'✅ SUCCESS' if self.phase_results['validation'] and self.phase_results['validation']['success'] else '❌ FAILED'}")
        
        if all_phases_success:
            print(f"\n🎉 PHASE 1.2 OVERALL: ✅ SUCCESS!")
            print(f"📁 Dataset ready for Phase 1.3 (Empirical Weight Optimization)")
        else:
            print(f"\n⚠️  PHASE 1.2 OVERALL: ❌ PARTIAL SUCCESS")
            print(f"📋 Check individual phase results for issues")
        
        print(f"\n📁 Complete results saved: {results_path}")
        
        return {
            'overall_success': all_phases_success,
            'summary_metrics': summary_metrics,
            'phase_results': self.phase_results
        }
    
    def run_complete_phase_1_2(self):
        """Execute complete Phase 1.2 process"""
        print(f"🚀 Starting Complete Phase 1.2 Process...")
        
        # Step 1: Clean Image Collection
        collection_success = self.execute_collection_phase()
        
        if not collection_success:
            print(f"\n⚠️  Collection phase issues detected, but continuing...")
        
        # Step 2: Noise Generation (proceed even with partial collection)
        generation_success = self.execute_generation_phase()
        
        if not generation_success:
            print(f"\n⚠️  Generation phase issues detected, but continuing to validation...")
        
        # Step 3: Dataset Validation
        validation_success = self.execute_validation_phase()
        
        # Step 4: Generate Summary
        final_summary = self.generate_phase_summary()
        
        return final_summary
    
    def quick_status_check(self):
        """Quick status check without full processing"""
        print(f"📊 QUICK DATASET STATUS CHECK")
        print(f"=" * 40)
        
        # Check current state
        status = self.collector.get_collection_status()
        
        print(f"📁 Current Dataset Status:")
        print(f"   Clean Images: {status['total_clean']:,}")
        print(f"   Noisy Images: {status['total_noisy']:,}")
        print(f"   Collection Progress: {status['completion_percentage']:.1f}%")
        
        # Check if validation report exists
        validation_report_path = self.dataset_dir / "metadata" / "validation_report.json"
        if validation_report_path.exists():
            try:
                with open(validation_report_path, 'r') as f:
                    report = json.load(f)
                print(f"   Last Validation Score: {report['overall_validation_score']:.3f}")
                print(f"   Last Validation: {'PASSED' if report['validation_passed'] else 'FAILED'}")
            except:
                print(f"   Last Validation: Report exists but couldn't parse")
        else:
            print(f"   Last Validation: Not run yet")
        
        return status

def main():
    """Main execution function"""
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        # Quick status check
        coordinator = Phase12Coordinator()
        coordinator.quick_status_check()
    else:
        # Full Phase 1.2 execution
        coordinator = Phase12Coordinator()
        summary = coordinator.run_complete_phase_1_2()
        
        # Return appropriate exit code
        sys.exit(0 if summary['overall_success'] else 1)

if __name__ == "__main__":
    main()
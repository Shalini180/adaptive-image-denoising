"""
Minimal Adaptive Process Visualizer
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class AdaptiveProcessVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_complete_process(self, image, noise_type, noise_level, save_plots=True):
        """Create a minimal visualization"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(image)
            axes[1].set_title(f'Noise: {noise_type}')
            axes[1].axis('off')
            
            if save_plots:
                plt.savefig(self.output_dir / f'process_{noise_type}.png')
            
            return fig, {'status': 'completed'}
        except Exception as e:
            print(f"Visualization error: {e}")
            return None, {'status': 'failed'}
    
    def visualize_refinement_iterations(self, image, noise_type, noise_level, max_iterations=2):
        """Create minimal refinement visualization"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.imshow(image)
            ax.set_title(f'Refinement: {noise_type}')
            ax.axis('off')
            
            plt.savefig(self.output_dir / f'refinement_{noise_type}.png')
            return fig, {'status': 'completed'}
        except Exception as e:
            print(f"Refinement visualization error: {e}")
            return None, {'status': 'failed'}

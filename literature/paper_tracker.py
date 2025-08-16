import pandas as pd
from datetime import datetime

def create_paper_database():
    """Create comprehensive database of 25 core denoising papers"""
    
    papers = [
        # Classical Methods (8 papers)
        {
            'id': 1,
            'title': 'A non-local algorithm for image denoising',
            'authors': 'Buades, A., Coll, B., Morel, J.M.',
            'year': 2005,
            'venue': 'CVPR',
            'category': 'Classical',
            'algorithm': 'Non-local means',
            'noise_types': 'Gaussian',
            'key_contribution': 'Non-local averaging based on patch similarity',
            'implementation_complexity': 'Medium',
            'computational_cost': 'High',
            'edge_preservation': 'Excellent',
            'psnr_improvement': '2-4 dB',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 2,
            'title': 'Image denoising by sparse 3-D transform-domain collaborative filtering',
            'authors': 'Dabov, K., Foi, A., Katkovnik, V., Egiazarian, K.',
            'year': 2007,
            'venue': 'IEEE TIP',
            'category': 'Classical',
            'algorithm': 'BM3D',
            'noise_types': 'Gaussian',
            'key_contribution': '3D collaborative filtering in transform domain',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High',
            'edge_preservation': 'Excellent',
            'psnr_improvement': '3-5 dB',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 3,
            'title': 'Bilateral filtering for gray and color images',
            'authors': 'Tomasi, C., Manduchi, R.',
            'year': 1998,
            'venue': 'ICCV',
            'category': 'Classical',
            'algorithm': 'Bilateral Filter',
            'noise_types': 'Gaussian',
            'key_contribution': 'Edge-preserving smoothing with spatial and intensity domains',
            'implementation_complexity': 'Easy',
            'computational_cost': 'Low',
            'edge_preservation': 'Good',
            'psnr_improvement': '1-3 dB',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 4,
            'title': 'Scale-space and edge detection using anisotropic diffusion',
            'authors': 'Perona, P., Malik, J.',
            'year': 1990,
            'venue': 'IEEE PAMI',
            'category': 'Classical',
            'algorithm': 'Anisotropic Diffusion',
            'noise_types': 'Gaussian',
            'key_contribution': 'Edge-preserving diffusion process',
            'implementation_complexity': 'Medium',
            'computational_cost': 'Medium',
            'edge_preservation': 'Excellent',
            'psnr_improvement': '2-4 dB',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 5,
            'title': 'Nonlinear total variation based noise removal algorithms',
            'authors': 'Rudin, L.I., Osher, S., Fatemi, E.',
            'year': 1992,
            'venue': 'Physica D',
            'category': 'Classical',
            'algorithm': 'Total Variation',
            'noise_types': 'Gaussian, Impulse',
            'key_contribution': 'Variational approach preserving edges',
            'implementation_complexity': 'Hard',
            'computational_cost': 'High',
            'edge_preservation': 'Excellent',
            'psnr_improvement': '2-5 dB',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 6,
            'title': 'Digital image enhancement and noise filtering by use of local statistics',
            'authors': 'Lee, J.S.',
            'year': 1980,
            'venue': 'IEEE PAMI',
            'category': 'Classical',
            'algorithm': 'Lee Filter',
            'noise_types': 'Speckle',
            'key_contribution': 'Adaptive filtering based on local statistics',
            'implementation_complexity': 'Medium',
            'computational_cost': 'Medium',
            'edge_preservation': 'Good',
            'psnr_improvement': '3-6 dB (speckle)',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 7,
            'title': 'Extrapolation, interpolation, and smoothing of stationary time series',
            'authors': 'Wiener, N.',
            'year': 1949,
            'venue': 'MIT Press',
            'category': 'Classical',
            'algorithm': 'Wiener Filter',
            'noise_types': 'Gaussian, Uniform',
            'key_contribution': 'Optimal linear filter minimizing MSE',
            'implementation_complexity': 'Hard',
            'computational_cost': 'Medium',
            'edge_preservation': 'Poor',
            'psnr_improvement': '2-4 dB',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 8,
            'title': 'The transformation of Poisson, binomial and negative-binomial data',
            'authors': 'Anscombe, F.J.',
            'year': 1948,
            'venue': 'Biometrika',
            'category': 'Classical',
            'algorithm': 'Anscombe Transform',
            'noise_types': 'Poisson',
            'key_contribution': 'Variance stabilizing transformation',
            'implementation_complexity': 'Easy',
            'computational_cost': 'Low',
            'edge_preservation': 'Depends on follow-up filter',
            'psnr_improvement': '1-3 dB',
            'status': 'To Review',
            'priority': 'Medium'
        },
        
        # Deep Learning Era (10 papers)
        {
            'id': 9,
            'title': 'Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising',
            'authors': 'Zhang, K., Zuo, W., Chen, Y., Meng, D., Zhang, L.',
            'year': 2017,
            'venue': 'IEEE TIP',
            'category': 'Deep Learning',
            'algorithm': 'DnCNN',
            'noise_types': 'Gaussian, JPEG artifacts',
            'key_contribution': 'Residual learning for noise prediction',
            'implementation_complexity': 'Medium',
            'computational_cost': 'Medium (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': '0.4-0.7 dB over BM3D',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 10,
            'title': 'MemNet: A persistent memory network for image restoration',
            'authors': 'Tai, Y., Yang, J., Liu, X., Xu, C.',
            'year': 2017,
            'venue': 'ICCV',
            'category': 'Deep Learning',
            'algorithm': 'MemNet',
            'noise_types': 'Gaussian',
            'key_contribution': 'Memory blocks for long-term dependencies',
            'implementation_complexity': 'Hard',
            'computational_cost': 'High (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': '0.3-0.6 dB over DnCNN',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 11,
            'title': 'Restormer: Efficient transformer for high-resolution image restoration',
            'authors': 'Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.',
            'year': 2022,
            'venue': 'CVPR',
            'category': 'Deep Learning',
            'algorithm': 'Restormer',
            'noise_types': 'Multiple',
            'key_contribution': 'Multi-scale transformer for restoration',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High (with GPU)',
            'edge_preservation': 'Excellent',
            'psnr_improvement': '0.5-1.0 dB over previous SOTA',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 12,
            'title': 'Toward convolutional blind denoising of real photographs',
            'authors': 'Guo, S., Yan, Z., Zhang, K., Zuo, W., Zhang, L.',
            'year': 2019,
            'venue': 'CVPR',
            'category': 'Deep Learning',
            'algorithm': 'CBDNet',
            'noise_types': 'Real-world noise',
            'key_contribution': 'Realistic noise modeling and removal',
            'implementation_complexity': 'Hard',
            'computational_cost': 'High (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': 'Significant on real images',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 13,
            'title': 'Variational denoising network: Toward blind noise modeling and removal',
            'authors': 'Yue, Z., Yong, H., Zhao, Q., Meng, D., Zhang, L.',
            'year': 2019,
            'venue': 'NeurIPS',
            'category': 'Deep Learning',
            'algorithm': 'VDN',
            'noise_types': 'Unknown noise',
            'key_contribution': 'Variational inference for blind denoising',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': 'Robust across noise types',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 14,
            'title': 'SwinIR: Image restoration using swin transformer',
            'authors': 'Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.',
            'year': 2021,
            'venue': 'ICCV',
            'category': 'Deep Learning',
            'algorithm': 'SwinIR',
            'noise_types': 'Multiple',
            'key_contribution': 'Swin transformer for image restoration',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High (with GPU)',
            'edge_preservation': 'Excellent',
            'psnr_improvement': '0.3-0.8 dB improvement',
            'status': 'To Review',
            'priority': 'Low'
        },
        {
            'id': 15,
            'title': 'NAFNet: Nonlinear activation free network for image restoration',
            'authors': 'Chen, L., Chu, X., Zhang, X., Sun, J.',
            'year': 2022,
            'venue': 'ECCV',
            'category': 'Deep Learning',
            'algorithm': 'NAFNet',
            'noise_types': 'Multiple',
            'key_contribution': 'Simplified architecture without nonlinear activations',
            'implementation_complexity': 'Hard',
            'computational_cost': 'High (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': 'Competitive with complex methods',
            'status': 'To Review',
            'priority': 'Low'
        },
        {
            'id': 16,
            'title': 'Multi-level wavelet-CNN for image restoration',
            'authors': 'Liu, P., Zhang, H., Zhang, K., Lin, L., Zuo, W.',
            'year': 2018,
            'venue': 'CVPR',
            'category': 'Deep Learning',
            'algorithm': 'MWCNN',
            'noise_types': 'Gaussian, JPEG',
            'key_contribution': 'Wavelet transform integration with CNN',
            'implementation_complexity': 'Hard',
            'computational_cost': 'High (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': '0.2-0.5 dB over DnCNN',
            'status': 'To Review',
            'priority': 'Low'
        },
        {
            'id': 17,
            'title': 'Uformer: A general u-shaped transformer for image restoration',
            'authors': 'Wang, Z., Cun, X., Bao, J., Zhou, W., Liu, J., Li, H.',
            'year': 2022,
            'venue': 'CVPR',
            'category': 'Deep Learning',
            'algorithm': 'Uformer',
            'noise_types': 'Multiple',
            'key_contribution': 'U-shaped transformer architecture',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': 'Competitive performance',
            'status': 'To Review',
            'priority': 'Low'
        },
        {
            'id': 18,
            'title': 'Simple baselines for image restoration',
            'authors': 'Chen, L., Lu, X., Zhang, J., Chu, X., Chen, C.',
            'year': 2022,
            'venue': 'ECCV',
            'category': 'Deep Learning',
            'algorithm': 'NAFNet variants',
            'noise_types': 'Multiple',
            'key_contribution': 'Simplified effective baselines',
            'implementation_complexity': 'Medium',
            'computational_cost': 'Medium (with GPU)',
            'edge_preservation': 'Good',
            'psnr_improvement': 'Strong baseline performance',
            'status': 'To Review',
            'priority': 'Medium'
        },
        
        # Adaptive & Multi-Noise Methods (7 papers)
        {
            'id': 19,
            'title': 'Optimal inversion of the Anscombe transformation in low-count Poisson image denoising',
            'authors': 'Foi, A., Trimeche, M., Katkovnik, V., Egiazarian, K.',
            'year': 2009,
            'venue': 'IEEE TIP',
            'category': 'Adaptive',
            'algorithm': 'Anscombe + BM3D',
            'noise_types': 'Poisson',
            'key_contribution': 'Optimal Anscombe transformation inversion',
            'implementation_complexity': 'Medium',
            'computational_cost': 'High',
            'edge_preservation': 'Excellent',
            'psnr_improvement': '2-4 dB for Poisson',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 20,
            'title': 'An analysis and implementation of the BM3D image denoising method',
            'authors': 'Lebrun, M., Buades, A., Morel, J.M.',
            'year': 2012,
            'venue': 'IPOL',
            'category': 'Adaptive',
            'algorithm': 'BM3D Analysis',
            'noise_types': 'Gaussian',
            'key_contribution': 'Detailed analysis and implementation guide',
            'implementation_complexity': 'Hard',
            'computational_cost': 'High',
            'edge_preservation': 'Excellent',
            'psnr_improvement': 'Reference implementation',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 21,
            'title': 'Optimal inversion of the generalized Anscombe transformation',
            'authors': 'Mäkitalo, M., Foi, A.',
            'year': 2013,
            'venue': 'IEEE TIP',
            'category': 'Adaptive',
            'algorithm': 'Generalized Anscombe',
            'noise_types': 'Poisson-Gaussian mix',
            'key_contribution': 'Handles mixed noise models',
            'implementation_complexity': 'Hard',
            'computational_cost': 'Medium',
            'edge_preservation': 'Good',
            'psnr_improvement': '1-3 dB for mixed noise',
            'status': 'To Review',
            'priority': 'High'
        },
        {
            'id': 22,
            'title': 'Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration',
            'authors': 'Chen, Y., Pock, T.',
            'year': 2017,
            'venue': 'CVPR',
            'category': 'Adaptive',
            'algorithm': 'TNRD',
            'noise_types': 'Multiple',
            'key_contribution': 'Trainable reaction-diffusion process',
            'implementation_complexity': 'Hard',
            'computational_cost': 'Medium',
            'edge_preservation': 'Good',
            'psnr_improvement': 'Competitive with CNN methods',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 23,
            'title': 'Connecting image denoising and high-level vision tasks via deep learning',
            'authors': 'Guo, S., Yan, Z., Zhang, K., Zuo, W., Zhang, L.',
            'year': 2020,
            'venue': 'IEEE TPAMI',
            'category': 'Adaptive',
            'algorithm': 'Task-aware denoising',
            'noise_types': 'Real-world',
            'key_contribution': 'Denoising for downstream tasks',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High (with GPU)',
            'edge_preservation': 'Task-dependent',
            'psnr_improvement': 'Task-specific metrics',
            'status': 'To Review',
            'priority': 'Low'
        },
        {
            'id': 24,
            'title': 'Multi-stage progressive image restoration',
            'authors': 'Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H., Shao, L.',
            'year': 2021,
            'venue': 'CVPR',
            'category': 'Adaptive',
            'algorithm': 'MPRNet',
            'noise_types': 'Multiple',
            'key_contribution': 'Multi-stage progressive restoration',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High (with GPU)',
            'edge_preservation': 'Excellent',
            'psnr_improvement': 'State-of-art on multiple tasks',
            'status': 'To Review',
            'priority': 'Medium'
        },
        {
            'id': 25,
            'title': 'Learning enriched features for real image restoration and enhancement',
            'authors': 'Zamir, S.W., Arora, A., Khan, S., Hayat, M., Khan, F.S., Yang, M.H.',
            'year': 2020,
            'venue': 'ECCV',
            'category': 'Adaptive',
            'algorithm': 'MIRNet',
            'noise_types': 'Real-world',
            'key_contribution': 'Multi-scale residual blocks',
            'implementation_complexity': 'Very Hard',
            'computational_cost': 'Very High (with GPU)',
            'edge_preservation': 'Excellent',
            'psnr_improvement': 'Strong real-world performance',
            'status': 'To Review',
            'priority': 'Medium'
        }
    ]
    
    df = pd.DataFrame(papers)
    df.to_csv('literature/paper_database.csv', index=False)
    print(f"Paper database created with {len(papers)} papers!")
    print(f"Categories: {df['category'].value_counts().to_dict()}")
    print(f"Saved to: literature/paper_database.csv")
    return df

def analyze_paper_distribution():
    """Analyze the distribution of papers across categories"""
    df = pd.read_csv('literature/paper_database.csv')
    
    print("\n=== PAPER DISTRIBUTION ANALYSIS ===")
    print(f"Total papers: {len(df)}")
    print(f"\nBy Category:")
    print(df['category'].value_counts())
    print(f"\nBy Implementation Complexity:")
    print(df['implementation_complexity'].value_counts())
    print(f"\nBy Priority:")
    print(df['priority'].value_counts())
    print(f"\nBy Noise Types (most common):")
    noise_counts = df['noise_types'].value_counts()
    print(noise_counts.head(10))

if __name__ == "__main__":
    # Create the database
    paper_df = create_paper_database()
    
    # Analyze distribution
    analyze_paper_distribution()
    
    print("\n✅ Paper database creation complete!")
    print("Next: Run algorithm analysis framework")
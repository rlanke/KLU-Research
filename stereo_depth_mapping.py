"""
Stereo Matching and Depth Map Generation for Underwater Shrimp Video
Implements multiple depth estimation approaches:
1. Pseudo-stereo from sequential frames (temporal stereo)
2. Block matching (BM) algorithm
3. Semi-Global Block Matching (SGBM)
4. Monocular depth estimation cues
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, List
import os

class StereoDepthEstimator:
    """
    Stereo matching and depth map generation for underwater video
    """
    
    def __init__(self):
        # Initialize stereo matchers
        self.stereo_bm = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        
        # SGBM parameters - optimized for underwater
        self.stereo_sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*6,  # Must be divisible by 16
            blockSize=11,
            P1=8 * 3 * 11**2,
            P2=32 * 3 * 11**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Post-processing parameters
        self.use_wls = False  # WLS filter not available in this OpenCV build
        
    def create_pseudo_stereo_pair(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create pseudo-stereo pair from sequential frames
        Simulates left-right camera displacement using temporal information
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        return gray1, gray2
    
    def compute_disparity_bm(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Compute disparity map using Block Matching algorithm
        """
        disparity = self.stereo_bm.compute(left, right)
        # Normalize to 8-bit
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return disparity_normalized
    
    def compute_disparity_sgbm(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Compute disparity map using Semi-Global Block Matching
        """
        disparity = self.stereo_sgbm.compute(left, right)
        
        # Post-process without WLS filter
        # Apply median filter for noise reduction
        disparity_float = disparity.astype(np.float32) / 16.0  # SGBM returns 16x disparity
        disparity_filtered = cv2.medianBlur(disparity_float.astype(np.uint8), 5)
        
        # Apply bilateral filter for edge-preserving smoothing
        disparity_filtered = cv2.bilateralFilter(disparity_filtered, 5, 50, 50)
        
        # Normalize
        disparity_normalized = cv2.normalize(disparity_filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return disparity_normalized
    
    def disparity_to_depth(self, disparity: np.ndarray, focal_length: float = 500, 
                          baseline: float = 0.1) -> np.ndarray:
        """
        Convert disparity to depth map
        Depth = (focal_length * baseline) / disparity
        """
        # Avoid division by zero
        disparity_float = disparity.astype(np.float32)
        disparity_float[disparity_float == 0] = 0.1
        
        # Calculate depth
        depth = (focal_length * baseline) / disparity_float
        
        # Clip unrealistic depth values
        depth = np.clip(depth, 0, 10)  # 0-10 meters range
        
        return depth
    
    def compute_monocular_depth_cues(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth using monocular cues:
        - Defocus blur
        - Atmospheric scattering
        - Image gradient
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Defocus blur estimation (higher blur = further away)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_measure = np.abs(laplacian)
        blur_measure = cv2.GaussianBlur(blur_measure, (15, 15), 0)
        blur_depth = 255 - cv2.normalize(blur_measure, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # 2. Atmospheric scattering (darker = closer in underwater)
        intensity_depth = 255 - gray
        
        # 3. Gradient magnitude (sharper edges = closer)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_depth = 255 - cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Combine cues with weights
        combined_depth = (
            0.3 * blur_depth.astype(np.float32) +
            0.4 * intensity_depth.astype(np.float32) +
            0.3 * gradient_depth.astype(np.float32)
        )
        
        combined_depth = np.clip(combined_depth, 0, 255).astype(np.uint8)
        
        # Apply bilateral filter for smoothing while preserving edges
        combined_depth = cv2.bilateralFilter(combined_depth, 9, 75, 75)
        
        return combined_depth
    
    def colorize_depth_map(self, depth_map: np.ndarray, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
        """
        Apply colormap to depth map for visualization
        """
        depth_colored = cv2.applyColorMap(depth_map, colormap)
        return depth_colored
    
    def create_3d_point_cloud(self, depth_map: np.ndarray, rgb_image: np.ndarray,
                             focal_length: float = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D point cloud from depth map
        Returns: points (N, 3) and colors (N, 3)
        """
        h, w = depth_map.shape
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Convert depth to actual distance
        depth_float = depth_map.astype(np.float32) / 255.0 * 10.0  # Scale to 0-10 meters
        
        # Compute 3D coordinates
        z = depth_float
        x_3d = (x - w/2) * z / focal_length
        y_3d = (y - h/2) * z / focal_length
        
        # Stack coordinates
        points = np.stack([x_3d, y_3d, z], axis=-1).reshape(-1, 3)
        
        # Get corresponding colors
        colors = rgb_image.reshape(-1, 3) / 255.0
        
        # Filter out invalid points
        valid_mask = (z.reshape(-1) > 0) & (z.reshape(-1) < 10)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        return points, colors


class DepthVideoProcessor:
    """
    Process entire video to generate depth maps
    """
    
    def __init__(self):
        self.estimator = StereoDepthEstimator()
        
    def process_video(self, video_path: str, output_path: str = None, 
                     method: str = 'all') -> List:
        """
        Process video and generate depth maps
        
        Methods:
        - 'temporal_stereo': Pseudo-stereo from sequential frames
        - 'monocular': Monocular depth cues
        - 'all': Both methods
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        print(f"Method: {method}")
        
        # Setup output video writer if path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width*3, height))
        
        frame_count = 0
        prev_frame = None
        sample_results = []
        sample_interval = max(1, total_frames // 12)  # Sample 12 frames
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            depth_maps = {}
            
            # Method 1: Temporal stereo (if we have previous frame)
            if method in ['temporal_stereo', 'all'] and prev_frame is not None:
                left, right = self.estimator.create_pseudo_stereo_pair(prev_frame, frame)
                
                # Compute disparity using SGBM (better quality)
                disparity_sgbm = self.estimator.compute_disparity_sgbm(left, right)
                depth_sgbm = self.estimator.disparity_to_depth(disparity_sgbm)
                depth_colored_sgbm = self.estimator.colorize_depth_map(
                    cv2.normalize(depth_sgbm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                    cv2.COLORMAP_TURBO
                )
                depth_maps['temporal_stereo'] = depth_colored_sgbm
            
            # Method 2: Monocular depth cues
            if method in ['monocular', 'all']:
                monocular_depth = self.estimator.compute_monocular_depth_cues(frame)
                depth_colored_mono = self.estimator.colorize_depth_map(
                    monocular_depth,
                    cv2.COLORMAP_TURBO
                )
                depth_maps['monocular'] = depth_colored_mono
            
            # Create visualization
            if depth_maps:
                if method == 'all' and 'temporal_stereo' in depth_maps:
                    combined = np.hstack([frame, depth_maps['temporal_stereo'], depth_maps['monocular']])
                elif 'temporal_stereo' in depth_maps:
                    combined = np.hstack([frame, depth_maps['temporal_stereo'], depth_maps['temporal_stereo']])
                else:
                    combined = np.hstack([frame, depth_maps['monocular'], depth_maps['monocular']])
                
                if output_path:
                    out.write(combined)
            
            # Save sample frames
            if frame_count % sample_interval == 0 and depth_maps:
                sample_results.append({
                    'frame_num': frame_count,
                    'original': frame.copy(),
                    'depth_maps': {k: v.copy() for k, v in depth_maps.items()},
                    'monocular_raw': monocular_depth.copy() if 'monocular' in depth_maps else None
                })
            
            prev_frame = frame.copy()
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        
        # Create visualizations
        self._create_visualizations(sample_results, method)
        
        return sample_results
    
    def _create_visualizations(self, sample_results: List, method: str):
        """
        Create comprehensive visualizations of depth estimation results
        """
        if not sample_results:
            return
        
        # Visualization 1: Sample frames with depth maps
        num_samples = min(6, len(sample_results))
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample in enumerate(sample_results[:num_samples]):
            # Original frame
            axes[idx, 0].imshow(cv2.cvtColor(sample['original'], cv2.COLOR_BGR2RGB))
            axes[idx, 0].set_title(f"Frame {sample['frame_num']}: Original")
            axes[idx, 0].axis('off')
            
            # Monocular depth
            if 'monocular' in sample['depth_maps']:
                axes[idx, 1].imshow(cv2.cvtColor(sample['depth_maps']['monocular'], cv2.COLOR_BGR2RGB))
                axes[idx, 1].set_title("Monocular Depth")
                axes[idx, 1].axis('off')
            
            # Temporal stereo depth
            if 'temporal_stereo' in sample['depth_maps']:
                axes[idx, 2].imshow(cv2.cvtColor(sample['depth_maps']['temporal_stereo'], cv2.COLOR_BGR2RGB))
                axes[idx, 2].set_title("Temporal Stereo Depth")
                axes[idx, 2].axis('off')
            
            # Combined/comparison
            if sample['monocular_raw'] is not None:
                axes[idx, 3].imshow(sample['monocular_raw'], cmap='turbo')
                axes[idx, 3].set_title("Depth Intensity Map")
                axes[idx, 3].axis('off')
                
                # Add colorbar
                im = axes[idx, 3].images[0]
                cbar = plt.colorbar(im, ax=axes[idx, 3], fraction=0.046, pad=0.04)
                cbar.set_label('Depth (closer → further)', rotation=270, labelpad=15)
        
        plt.suptitle(f'Stereo Matching & Depth Map Generation - water-shrimp.mp4 ({method} method)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/depth_map_samples.png', dpi=150, bbox_inches='tight')
        print("Saved: depth_map_samples.png")
        plt.close()
        
        # Visualization 2: Comparison view
        if len(sample_results) >= 3:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            
            for idx in range(3):
                sample = sample_results[idx * (len(sample_results) // 3)]
                
                # Original
                axes[idx, 0].imshow(cv2.cvtColor(sample['original'], cv2.COLOR_BGR2RGB))
                axes[idx, 0].set_title(f"Frame {sample['frame_num']}")
                axes[idx, 0].axis('off')
                
                # Monocular depth
                if 'monocular' in sample['depth_maps']:
                    axes[idx, 1].imshow(cv2.cvtColor(sample['depth_maps']['monocular'], 
                                                     cv2.COLOR_BGR2RGB))
                    axes[idx, 1].set_title("Monocular Depth Map")
                    axes[idx, 1].axis('off')
                
                # Temporal stereo or monocular raw
                if 'temporal_stereo' in sample['depth_maps']:
                    axes[idx, 2].imshow(cv2.cvtColor(sample['depth_maps']['temporal_stereo'],
                                                     cv2.COLOR_BGR2RGB))
                    axes[idx, 2].set_title("Temporal Stereo Depth")
                elif sample['monocular_raw'] is not None:
                    axes[idx, 2].imshow(sample['monocular_raw'], cmap='turbo')
                    axes[idx, 2].set_title("Depth Intensity")
                axes[idx, 2].axis('off')
            
            plt.suptitle('Depth Estimation: Progressive Samples',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('/mnt/user-data/outputs/depth_comparison_grid.png',
                        dpi=150, bbox_inches='tight')
            print("Saved: depth_comparison_grid.png")
            plt.close()
        
        # Visualization 3: 3D point cloud projection (sample frame)
        if sample_results:
            sample = sample_results[len(sample_results) // 2]
            if sample['monocular_raw'] is not None:
                self._create_3d_visualization(sample)


    def _create_3d_visualization(self, sample: dict):
        """
        Create 3D point cloud visualization
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        depth_map = sample['monocular_raw']
        rgb_image = sample['original']
        
        # Downsample for visualization
        scale = 4
        depth_small = cv2.resize(depth_map, (depth_map.shape[1]//scale, 
                                            depth_map.shape[0]//scale))
        rgb_small = cv2.resize(rgb_image, (rgb_image.shape[1]//scale,
                                          rgb_image.shape[0]//scale))
        
        # Generate point cloud
        points, colors = self.estimator.create_3d_point_cloud(
            depth_small, 
            cv2.cvtColor(rgb_small, cv2.COLOR_BGR2RGB)
        )
        
        # Sample points for visualization (too many points slow down rendering)
        sample_idx = np.random.choice(len(points), min(5000, len(points)), replace=False)
        points_sampled = points[sample_idx]
        colors_sampled = colors[sample_idx]
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 12))
        
        # View 1: Top view
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],
                   c=colors_sampled, s=1, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Depth')
        ax1.set_title('3D Point Cloud - Top View')
        ax1.view_init(elev=90, azim=-90)
        
        # View 2: Side view
        ax2 = fig.add_subplot(2, 2, 2, projection='3d')
        ax2.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],
                   c=colors_sampled, s=1, alpha=0.6)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Depth')
        ax2.set_title('3D Point Cloud - Side View')
        ax2.view_init(elev=0, azim=-90)
        
        # View 3: Perspective view
        ax3 = fig.add_subplot(2, 2, 3, projection='3d')
        ax3.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],
                   c=colors_sampled, s=1, alpha=0.6)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Depth')
        ax3.set_title('3D Point Cloud - Perspective View')
        ax3.view_init(elev=30, azim=45)
        
        # View 4: Another angle
        ax4 = fig.add_subplot(2, 2, 4, projection='3d')
        ax4.scatter(points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2],
                   c=colors_sampled, s=1, alpha=0.6)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Depth')
        ax4.set_title('3D Point Cloud - Alternate View')
        ax4.view_init(elev=20, azim=-60)
        
        plt.suptitle(f'3D Reconstruction from Depth Map - Frame {sample["frame_num"]}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/depth_3d_pointcloud.png',
                    dpi=150, bbox_inches='tight')
        print("Saved: depth_3d_pointcloud.png")
        plt.close()


def main():
    """
    Main execution
    """
    print("="*70)
    print("Stereo Matching & Depth Map Generation for Underwater Shrimp Video")
    print("="*70)
    print()
    
    # Initialize processor
    processor = DepthVideoProcessor()
    
    # Process video with all methods
    input_video = "/mnt/user-data/uploads/water-shrimp.mp4"
    output_video = "/mnt/user-data/outputs/water_shrimp_depth_maps.mp4"
    
    print("Processing with multiple depth estimation methods...")
    print("- Temporal stereo (pseudo-stereo from sequential frames)")
    print("- Monocular depth cues (defocus, scattering, gradients)")
    print()
    
    # Process video
    results = processor.process_video(input_video, output_video, method='all')
    
    print("\n" + "="*70)
    print("Depth map generation complete!")
    print(f"Output video: {output_video}")
    print("="*70)


if __name__ == "__main__":
    main()

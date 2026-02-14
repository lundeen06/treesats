"""
High-level satellite tracking pipeline interface.

This module provides the highest-level abstraction for satellite detection
and tracking, designed to integrate with other parts of the system.

Example:
    from sat.computer_vision.pipeline import SatelliteTracker

    # Initialize tracker
    tracker = SatelliteTracker()

    # Process video and get per-frame satellite positions
    for frame in tracker.stream_video("simulation.mp4"):
        # N×3 matrix: [sat_id, x_norm, y_norm]
        # X: -1 (left) to +1 (right)
        # Y: -1 (bottom) to +1 (top)
        positions = frame.normalized
        
        # Feed to next stage of pipeline
        process_satellite_positions(frame.frame_id, positions)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union
import numpy as np
import cv2

from .inference import load_model, get_device


@dataclass
class TrackingFrame:
    """
    Tracking results for a single frame.
    
    Attributes:
        frame_id: Frame index (0-based)
        pixel: N×3 numpy array of [sat_id, x_pixel, y_pixel]
        normalized: N×3 numpy array of [sat_id, x_norm, y_norm]
                   where x ∈ [-1, 1] (left to right)
                   and y ∈ [-1, 1] (bottom to top)
        image_size: Tuple of (width, height) in pixels
    """
    frame_id: int
    pixel: np.ndarray
    normalized: np.ndarray
    image_size: tuple
    
    @property
    def num_satellites(self) -> int:
        """Number of satellites detected in this frame."""
        return len(self.pixel)
    
    @property
    def satellite_ids(self) -> List[int]:
        """List of satellite IDs in this frame."""
        if len(self.pixel) == 0:
            return []
        return [int(row[0]) for row in self.pixel]
    
    def get_position(self, sat_id: int, normalized: bool = True) -> Optional[tuple]:
        """
        Get position of a specific satellite by ID.
        
        Args:
            sat_id: Satellite track ID
            normalized: If True, return normalized coords; else pixel coords
            
        Returns:
            Tuple of (x, y) or None if satellite not found
        """
        matrix = self.normalized if normalized else self.pixel
        for row in matrix:
            if int(row[0]) == sat_id:
                return (float(row[1]), float(row[2]))
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "frame_id": self.frame_id,
            "num_satellites": self.num_satellites,
            "image_size": self.image_size,
            "pixel": self.pixel.tolist(),
            "normalized": self.normalized.tolist(),
        }


class SatelliteTracker:
    """
    High-level interface for satellite tracking.
    
    This class provides the main entry point for processing simulation
    videos and extracting satellite positions with persistent track IDs.
    
    Example:
        tracker = SatelliteTracker()
        
        # Stream processing (memory efficient)
        for frame in tracker.stream_video("video.mp4"):
            print(f"Frame {frame.frame_id}: {frame.num_satellites} satellites")
            print(frame.normalized)  # N×3 matrix
        
        # Batch processing
        frames = tracker.process_video("video.mp4")
        
        # Single frame
        result = tracker.process_frame(image_array)
    """
    
    def __init__(
        self,
        weights_path: str = None,
        conf_threshold: float = 0.25,
        tracker: str = "botsort.yaml",
    ):
        """
        Initialize the satellite tracker.
        
        Args:
            weights_path: Path to model weights (default: uses best.pt)
            conf_threshold: Detection confidence threshold (0-1)
            tracker: Tracking algorithm - "botsort.yaml" or "bytetrack.yaml"
        """
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.tracker = tracker
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = load_model(self.weights_path)
        return self._model
    
    def stream_video(
        self,
        video_path: str,
        conf_threshold: float = None,
    ) -> Iterator[TrackingFrame]:
        """
        Stream satellite tracking results frame-by-frame.
        
        This is the recommended method for processing videos as it's
        memory efficient - only one frame is held in memory at a time.
        
        Args:
            video_path: Path to video file
            conf_threshold: Override default confidence threshold
            
        Yields:
            TrackingFrame for each frame in the video
            
        Example:
            tracker = SatelliteTracker()
            for frame in tracker.stream_video("simulation.mp4"):
                # Process each frame
                positions = frame.normalized  # N×3 [id, x, y]
                feed_to_pipeline(frame.frame_id, positions)
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # Get video properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        image_size = (width, height)
        
        # Run tracking with streaming
        results = self.model.track(
            source=str(video_path),
            conf=conf,
            tracker=self.tracker,
            save=False,
            show=False,
            imgsz=256,
            device=get_device(),
            persist=True,
            stream=True,
            verbose=False,
        )
        
        frame_id = 0
        for result in results:
            yield self._process_result(result, frame_id, image_size)
            frame_id += 1
    
    def process_video(
        self,
        video_path: str,
        conf_threshold: float = None,
    ) -> List[TrackingFrame]:
        """
        Process entire video and return all frames.
        
        Note: For large videos, use stream_video() instead to avoid
        loading all results into memory at once.
        
        Args:
            video_path: Path to video file
            conf_threshold: Override default confidence threshold
            
        Returns:
            List of TrackingFrame objects, one per frame
        """
        return list(self.stream_video(video_path, conf_threshold))
    
    def process_frame(
        self,
        image: np.ndarray,
        conf_threshold: float = None,
    ) -> TrackingFrame:
        """
        Process a single frame/image.
        
        Note: For video sequences, use stream_video() which maintains
        persistent track IDs across frames. This method treats each
        call as independent (no tracking between calls).
        
        Args:
            image: Input image as numpy array (BGR format)
            conf_threshold: Override default confidence threshold
            
        Returns:
            TrackingFrame with detection results
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        height, width = image.shape[:2]
        image_size = (width, height)
        
        # Run detection (not tracking for single frame)
        results = self.model.predict(
            source=image,
            conf=conf,
            save=False,
            show=False,
            imgsz=256,
            device=get_device(),
            verbose=False,
        )
        
        if results:
            return self._process_result(results[0], 0, image_size, use_tracking=False)
        else:
            return TrackingFrame(
                frame_id=0,
                pixel=np.empty((0, 3), dtype=np.float32),
                normalized=np.empty((0, 3), dtype=np.float32),
                image_size=image_size,
            )
    
    def _process_result(
        self,
        result,
        frame_id: int,
        image_size: tuple,
        use_tracking: bool = True,
    ) -> TrackingFrame:
        """Convert YOLO result to TrackingFrame."""
        width, height = image_size
        
        pixel_rows = []
        normalized_rows = []
        
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            has_ids = use_tracking and boxes.id is not None
            
            for i, box in enumerate(boxes):
                # Get track ID (or use index if no tracking)
                if has_ids:
                    track_id = int(boxes.id[i])
                else:
                    track_id = i
                
                # Get center position
                xywh = box.xywh[0].tolist()
                x_center = xywh[0]
                y_center = xywh[1]
                
                # Pixel coordinates
                pixel_rows.append([track_id, x_center, y_center])
                
                # Normalized coordinates
                # X: -1 (left) to +1 (right)
                # Y: -1 (bottom) to +1 (top)
                x_norm = (x_center / (width / 2)) - 1.0
                y_norm = 1.0 - (y_center / (height / 2))
                
                normalized_rows.append([track_id, x_norm, y_norm])
        
        # Convert to numpy arrays
        if pixel_rows:
            pixel = np.array(pixel_rows, dtype=np.float32)
            normalized = np.array(normalized_rows, dtype=np.float32)
        else:
            pixel = np.empty((0, 3), dtype=np.float32)
            normalized = np.empty((0, 3), dtype=np.float32)
        
        return TrackingFrame(
            frame_id=frame_id,
            pixel=pixel,
            normalized=normalized,
            image_size=image_size,
        )
    
    def create_visualization(
        self,
        video_path: str,
        output_path: str = None,
        conf_threshold: float = None,
        font_scale: float = 0.25,
    ) -> str:
        """
        Create annotated video with tracking visualization.
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (default: auto-generated)
            conf_threshold: Override default confidence threshold
            font_scale: Font size for labels
            
        Returns:
            Path to output video
        """
        from .inference import track_video_custom
        
        return track_video_custom(
            video_path=video_path,
            output_path=output_path,
            weights_path=self.weights_path,
            conf_threshold=conf_threshold or self.conf_threshold,
            tracker=self.tracker,
            font_scale=font_scale,
        )


# Convenience function for quick usage
def track_video(video_path: str, conf_threshold: float = 0.25) -> Iterator[TrackingFrame]:
    """
    Convenience function to track satellites in a video.
    
    Args:
        video_path: Path to video file
        conf_threshold: Detection confidence threshold
        
    Yields:
        TrackingFrame for each frame
        
    Example:
        from sat.computer_vision.pipeline import track_video
        
        for frame in track_video("simulation.mp4"):
            print(frame.normalized)  # N×3 matrix [id, x, y]
    """
    tracker = SatelliteTracker(conf_threshold=conf_threshold)
    yield from tracker.stream_video(video_path)

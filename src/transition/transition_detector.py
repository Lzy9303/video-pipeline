import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
from skimage.metrics import structural_similarity as ssim
import os

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


@dataclass
class TransitionPoint:
    """Represents a detected transition point in a video"""
    frame_number: int  # Primary identifier - frame number
    timestamp: float  # Calculated from frame_number / fps
    score: float
    transition_type: str  # 'movement', 'scene_change', 'lighting_change'
    confidence: float  # 0.0 to 1.0


@dataclass
class VideoTransitionAnalysis:
    """Result of video transition analysis"""
    file_path: str
    success: bool
    transitions: List[TransitionPoint] = None
    total_frames: int = 0
    fps: float = 0.0
    duration: float = 0.0
    processing_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "file": self.file_path,
            "success": self.success,
            "transitions": [
                {
                    "frame_num": int(t.frame_number),
                    "type": t.transition_type,
                    "score": float(t.score)
                } for t in (self.transitions or [])
            ],
            "total_frames": int(self.total_frames),
            "fps": float(self.fps),
            "duration": float(self.duration),
            "processing_time": float(self.processing_time),
            "error": self.error
        }


class VideoTransitionDetector:
    """Detects transitions in videos based on frame analysis"""

    def __init__(self,
                 movement_threshold: float = 0.15,
                 scene_change_threshold: float = 0.3,
                 lighting_threshold: float = 0.2,
                 min_transition_gap: float = 0.1,
                 movement_history_size: int = 5):
        """
        Initialize the transition detector.

        Args:
            movement_threshold: Threshold for detecting movement-based transitions
            scene_change_threshold: Threshold for detecting scene changes
            lighting_threshold: Threshold for detecting lighting changes
            min_transition_gap: Minimum time (seconds) between transitions
            movement_history_size: Number of frames to track for baseline movement
        """
        self.movement_threshold = movement_threshold
        self.scene_change_threshold = scene_change_threshold
        self.lighting_threshold = lighting_threshold
        self.min_transition_gap = min_transition_gap
        self.movement_history_size = movement_history_size

    def _calculate_movement_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate movement score using frame difference and edge detection"""
        try:
            # Convert to grayscale if needed
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame

            if len(curr_frame.shape) == 3:
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = curr_frame

            # Resize to standard size for consistent analysis
            height, width = 240, 320  # Smaller for faster processing
            prev_resized = cv2.resize(prev_gray, (width, height))
            curr_resized = cv2.resize(curr_gray, (width, height))

            # Method 1: Frame difference
            frame_diff = cv2.absdiff(prev_resized, curr_resized)

            # Method 2: Edge-based movement detection
            # Apply Gaussian blur to reduce noise
            prev_blur = cv2.GaussianBlur(prev_resized, (5, 5), 0)
            curr_blur = cv2.GaussianBlur(curr_resized, (5, 5), 0)

            # Calculate edges using Canny
            prev_edges = cv2.Canny(prev_blur, 50, 150)
            curr_edges = cv2.Canny(curr_blur, 50, 150)

            # Edge movement difference
            edge_diff = cv2.absdiff(prev_edges, curr_edges)

            # Method 3: Optical flow with good feature points
            # Find corners for tracking
            corners = cv2.goodFeaturesToTrack(prev_blur, maxCorners=100,
                                            qualityLevel=0.01, minDistance=10)

            flow_magnitude = 0.0
            if corners is not None and len(corners) > 10:
                # Calculate optical flow
                next_corners, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_blur, curr_blur, corners, None,
                    winSize=(15, 15), maxLevel=2)

                if next_corners is not None:
                    # Calculate movement vectors for good points
                    good_old = corners[status == 1]
                    good_new = next_corners[status == 1]

                    if len(good_old) > 5:
                        # Calculate movement magnitude
                        movement_vectors = good_new - good_old
                        magnitudes = np.sqrt(movement_vectors[:, 0]**2 + movement_vectors[:, 1]**2)
                        flow_magnitude = float(np.mean(magnitudes))

            # Combine different methods for robust movement detection
            frame_movement = float(np.mean(frame_diff) / 255.0)
            edge_movement = float(np.mean(edge_diff) / 255.0)
            flow_movement = flow_magnitude / 10.0  # Normalize optical flow

            # Weighted combination emphasizing edge and optical flow changes
            combined_score = (frame_movement * 0.3 +
                            edge_movement * 0.4 +
                            flow_movement * 0.3)

            return float(combined_score)

        except Exception as e:
            logger.debug(f"Movement calculation failed: {e}")
            return 0.0

    def _calculate_frame_difference(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate 1 - SSIM as a scene difference metric (0..1)."""
        try:
            height, width = 480, 640
            prev_resized = cv2.resize(prev_frame, (width, height))
            curr_resized = cv2.resize(curr_frame, (width, height))

            if len(prev_resized.shape) == 3:
                prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_resized

            if len(curr_resized.shape) == 3:
                curr_gray = cv2.cvtColor(curr_resized, cv2.COLOR_BGR2GRAY)
            else:
                curr_gray = curr_resized

            ssim_val = ssim(prev_gray, curr_gray)
            return float(max(0.0, min(1.0, 1.0 - ssim_val)))
        except Exception as e:
            logger.debug(f"Frame difference calculation failed: {e}")
            return 0.0

    def _calculate_histogram_difference(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate histogram difference for lighting/color changes"""
        try:
            # Calculate histograms for each channel
            prev_hist = []
            curr_hist = []

            if len(prev_frame.shape) == 3:
                for i in range(3):  # BGR channels
                    prev_hist.append(cv2.calcHist([prev_frame], [i], None, [256], [0, 256]))
                    curr_hist.append(cv2.calcHist([curr_frame], [i], None, [256], [0, 256]))
            else:
                prev_hist.append(cv2.calcHist([prev_frame], [0], None, [256], [0, 256]))
                curr_hist.append(cv2.calcHist([curr_frame], [0], None, [256], [0, 256]))

            # Calculate correlation for each channel
            correlations = []
            for p_hist, c_hist in zip(prev_hist, curr_hist):
                correlation = cv2.compareHist(p_hist, c_hist, cv2.HISTCMP_CORREL)
                correlations.append(correlation)

            # Return average correlation (1 - correlation gives difference)
            avg_correlation = np.mean(correlations)
            return float(1.0 - avg_correlation)

        except Exception as e:
            logger.debug(f"Histogram calculation failed: {e}")
            return 0.0

    def _analyze_transition_type(self, movement_score: float, scene_score: float,
                                lighting_score: float) -> Tuple[str, float]:
        """Determine transition type and confidence based on scores"""
        scores = {
            'movement': movement_score,
            'scene_change': scene_score,
            'lighting_change': lighting_score
        }

        # Find the dominant transition type
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        # Calculate confidence based on how much the max score exceeds others
        other_scores = [v for k, v in scores.items() if k != max_type]
        avg_other = float(np.mean(other_scores)) if other_scores else 0.0

        confidence = float(min(1.0, (max_score - avg_other) / max_score)) if max_score > 0 else 0.0

        return max_type, confidence

    def _detect_sudden_movement_change(self, movement_history: List[float], current_movement: float,
                                     frame_number: int, timestamp: float) -> Tuple[bool, float]:
        """
        Detect sudden changes in movement patterns.

        Args:
            movement_history: List of recent movement scores
            current_movement: Current frame movement score
            frame_number: Current frame number
            timestamp: Current timestamp

        Returns:
            (is_sudden_change, change_magnitude)
        """
        if len(movement_history) < 3:
            return False, 0.0

        try:
            # Calculate baseline movement (average of recent history)
            baseline_movement = float(np.mean(movement_history[-3:]))

            # Calculate movement change from baseline
            movement_change = abs(current_movement - baseline_movement)

            # Method 1: Absolute threshold check
            threshold_exceeded = movement_change > self.movement_threshold

            # Method 2: Relative change check (sudden spike or drop)
            if baseline_movement > 0.01:  # Avoid division by zero
                relative_change = movement_change / baseline_movement
                relative_threshold_exceeded = relative_change > 2.0  # 200% change
            else:
                relative_threshold_exceeded = current_movement > self.movement_threshold

            # Method 3: Standard deviation check for outliers
            if len(movement_history) >= self.movement_history_size:
                std_dev = float(np.std(movement_history))
                mean_movement = float(np.mean(movement_history))
                z_score = abs(current_movement - mean_movement) / (std_dev + 1e-6)
                outlier_detected = z_score > 2.5  # 2.5 standard deviations
            else:
                outlier_detected = False

            # Determine if sudden change occurred
            is_sudden_change = threshold_exceeded or relative_threshold_exceeded or outlier_detected

            # Calculate change magnitude for confidence scoring
            change_magnitude = float(min(1.0, movement_change / (self.movement_threshold + 1e-6)))

            if is_sudden_change:
                logger.debug(f"Sudden movement detected at {timestamp:.2f}s: "
                           f"current={current_movement:.4f}, baseline={baseline_movement:.4f}, "
                           f"change={movement_change:.4f}, magnitude={change_magnitude:.4f}")

            return is_sudden_change, change_magnitude

        except Exception as e:
            logger.debug(f"Movement change detection failed: {e}")
            return False, 0.0

    def detect_transitions(self, video_path: str,
                          sample_rate: int = 1,
                          max_transitions: int = 50,
                          start_sec: Optional[float] = None,
                          end_sec: Optional[float] = None,
                          dump_frames: bool = True,
                          output_dir: Optional[str] = None) -> VideoTransitionAnalysis:
        """
        Detect transitions in a video file.

        Args:
            video_path: Path to the video file
            sample_rate: Process every Nth frame (higher = faster, less accurate)
            max_transitions: Maximum number of transitions to detect
            start_time: Optional start time (seconds) to begin analysis
            end_time: Optional end time (seconds) to stop analysis
            dump_frames: Whether to save before/after frames to disk (default True)
            output_dir: Optional directory to save frames (default: same directory as video)

        Returns:
            VideoTransitionAnalysis object with detected transitions
        """
        logger.info(f"Starting transition detection for: {video_path}")
        t0 = time.time()

        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return VideoTransitionAnalysis(
                    file_path=video_path,
                    success=False,
                    error=f"Could not open video file: {video_path}"
                )

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps and fps > 0 else 0

            # Determine frame bounds if a range is specified
            if fps and fps > 0:
                start_frame = int(max(0, round(((start_sec or 0.0)) * fps)))
                end_frame = int(min(max(0, total_frames - 1), round((end_sec * fps)) if end_sec is not None else total_frames - 1))
            else:
                start_frame = 0
                end_frame = max(0, total_frames - 1)

            logger.debug(f"Video properties - FPS: {fps}, Frames: {total_frames}, Duration: {duration:.2f}s")
            if start_sec is not None or end_sec is not None:
                logger.debug(f"Analyzing range: frames {start_frame}..{end_frame} (times {start_sec or 0:.2f}s..{end_sec if end_sec is not None else duration:.2f}s)")

            # Setup frame dumping directory
            dump_dir = None
            if dump_frames:
                if output_dir:
                    dump_dir = output_dir
                    os.makedirs(dump_dir, exist_ok=True)
                else:
                    dump_dir = os.path.dirname(video_path)
                logger.info(f"Frames will be saved to: {dump_dir}")

            transitions = []
            prev_frame = None
            prev_frame_number = None
            frame_count = 0
            last_transition_time = -self.min_transition_gap
            transition_idx = 0

            # Movement history tracking for sudden change detection
            movement_history = []

            logger.info(f"Processing every {sample_rate} frames...")

            while True:
                # Grab next frame (fast), decode only sampled frames
                ret = cap.grab()
                if not ret:
                    break

                frame_count += 1

                # Respect range: skip until start_frame
                if frame_count < start_frame:
                    continue

                if frame_count % sample_rate != 0:
                    continue

                ret, frame = cap.retrieve()
                if not ret:
                    break

                current_time = frame_count / fps

                # Stop if past end frame
                if frame_count > end_frame:
                    break

                if prev_frame is not None:
                    # Calculate different types of changes
                    movement_score = self._calculate_movement_score(prev_frame, frame)
                    scene_score = self._calculate_frame_difference(prev_frame, frame)
                    lighting_score = self._calculate_histogram_difference(prev_frame, frame)

                    # Add current movement to history
                    movement_history.append(movement_score)
                    if len(movement_history) > self.movement_history_size:
                        movement_history.pop(0)  # Keep only recent history

                    # Detect sudden movement changes
                    is_sudden_movement, movement_change_magnitude = self._detect_sudden_movement_change(
                        movement_history, movement_score, frame_count, current_time
                    )

                    logger.debug(f"Frame {frame_count} ({current_time:.2f}s) - "
                               f"Movement: {movement_score:.4f}, Scene: {scene_score:.3f}, "
                               f"Lighting: {lighting_score:.3f}, Sudden: {is_sudden_movement}")

                    # Check for different types of transitions
                    is_scene_change = scene_score > self.scene_change_threshold
                    is_lighting_change = lighting_score > self.lighting_threshold

                    # Check if any transition detected and minimum gap respected
                    if (is_sudden_movement or is_scene_change or is_lighting_change) and \
                       (current_time - last_transition_time) >= self.min_transition_gap:

                        # Determine primary transition type
                        if is_sudden_movement:
                            transition_type = "movement"
                            primary_score = movement_change_magnitude
                        elif is_scene_change:
                            transition_type = "scene_change"
                            primary_score = scene_score
                        elif is_lighting_change:
                            transition_type = "lighting_change"
                            primary_score = lighting_score
                        else:
                            transition_type = "unknown"
                            primary_score = max(movement_score, scene_score, lighting_score)

                        # Calculate confidence based on detection strength
                        if is_sudden_movement:
                            confidence = float(min(1.0, movement_change_magnitude))
                        else:
                            # Use the old method for scene/lighting changes
                            transition_type_old, confidence = self._analyze_transition_type(
                                movement_score, scene_score, lighting_score
                            )

                        transition = TransitionPoint(
                            frame_number=int(frame_count),
                            timestamp=float(current_time),
                            score=float(primary_score),
                            transition_type=transition_type,
                            confidence=float(confidence)
                        )

                        transitions.append(transition)
                        last_transition_time = current_time

                        logger.info(f"Transition detected at {current_time:.2f}s - "
                                  f"Type: {transition_type}, Score: {transition.score:.3f}, "
                                  f"Confidence: {confidence:.3f}")

                        # Dump frames before and after transition
                        if dump_frames and dump_dir and prev_frame is not None:
                            try:
                                # Save before frame (previous frame)
                                before_filename = os.path.join(dump_dir, f"transition_{transition_idx:03d}_before.jpg")
                                cv2.imwrite(before_filename, prev_frame)
                                logger.debug(f"Saved before frame: {before_filename}")

                                # Save after frame (current frame)
                                after_filename = os.path.join(dump_dir, f"transition_{transition_idx:03d}_after.jpg")
                                cv2.imwrite(after_filename, frame)
                                logger.debug(f"Saved after frame: {after_filename}")

                                transition_idx += 1
                            except Exception as e:
                                logger.error(f"Failed to dump frames for transition: {e}")

                        # Limit number of transitions if a positive cap is set
                        if max_transitions and max_transitions > 0 and len(transitions) >= max_transitions:
                            logger.info(f"Reached maximum transitions limit ({max_transitions})")
                            break

                prev_frame = frame.copy()

                # Progress reporting
                if frame_count % (sample_rate * 30) == 0:  # Every ~1 second of video
                    progress = (frame_count / total_frames) * 100
                    logger.debug(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

            cap.release()
            processing_time = time.time() - t0

            logger.info(f"Transition detection completed in {processing_time:.2f}s")
            logger.info(f"Found {len(transitions)} transitions")

            return VideoTransitionAnalysis(
                file_path=video_path,
                success=True,
                transitions=transitions,
                total_frames=int(total_frames),
                fps=float(fps),
                duration=float(duration),
                processing_time=float(processing_time)
            )

        except Exception as e:
            logger.error(f"Error during transition detection: {str(e)}", exc_info=True)
            return VideoTransitionAnalysis(
                file_path=video_path,
                success=False,
                error=f"Transition detection error: {str(e)}"
            )

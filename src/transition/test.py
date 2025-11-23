import os
import uuid
from pathlib import Path
from transition_detector import VideoTransitionDetector

# Setup paths
test_resources_dir = Path(__file__).parent / "test_resources"
test_result_dir = Path(__file__).parent / "test_result"
input_video = test_resources_dir / "input.mp4"

# Create test result directory
test_result_dir.mkdir(parents=True, exist_ok=True)

# Create unique output directory with UUID
output_uuid = str(uuid.uuid4())
output_dir = test_result_dir / output_uuid
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input video: {input_video}")
print(f"Output directory: {output_dir}")

# Check if input video exists
if not input_video.exists():
    print(f"Error: Input video not found at {input_video}")
    exit(1)

# Run transition detection
detector = VideoTransitionDetector(
    movement_threshold=0.25,
    scene_change_threshold=0.5,
    lighting_threshold=0.3,
    min_transition_gap=0.05
)

result = detector.detect_transitions(
    video_path=str(input_video),
    sample_rate=1,
    max_transitions=500,
    dump_frames=True,
    output_dir=str(output_dir)
)

# Print results
print(f"\nTransition Detection Results:")
print(f"Success: {result.success}")
print(f"Total frames: {result.total_frames}")
print(f"FPS: {result.fps}")
print(f"Duration: {result.duration:.2f}s")
print(f"Processing time: {result.processing_time:.2f}s")
print(f"Transitions found: {len(result.transitions) if result.transitions else 0}")

if result.transitions:
    print(f"\nTransitions:")
    for i, transition in enumerate(result.transitions):
        print(f"  {i+1}. Frame {transition.frame_number} @ {transition.timestamp:.2f}s - "
              f"Type: {transition.transition_type}, Score: {transition.score:.3f}, "
              f"Confidence: {transition.confidence:.3f}")

if result.error:
    print(f"Error: {result.error}")

print(f"\nFrames saved to: {output_dir}")

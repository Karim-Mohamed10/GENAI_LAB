from utils.video import read_video, save_video, VideoProcessor, VideoWriter
from tracker.keypoints_tracker import KeypointsTracker
import os
import torch
import cv2
import numpy as np

# ==============================================================================
# 1. FIELD CONNECTION MAP (Adjusted to your Hand-Drawn Map)
# ==============================================================================
# IDs 8 and 21 (Penalty Spots) are excluded from all tuples.
FIELD_CONNECTIONS = [
    # --- LEFT SIDE (As per your drawing) ---
    # Penalty Box (Big Box)
    (1,9),(9, 10), (10, 11), (11, 12), (12, 4), 
    #small box
    (2,6),(6,7),(7,3),(2,3),
    #outer left goal
    (0,1),(1,2),(3,4),(4,5),
    
    # --- RIGHT SIDE (As per your drawing) ---
    # Penalty Box (Big Box)
    (17, 18), (18, 19), (19, 20), (20, 28),(17,25),
    # Small Box (6-yard)
    (22, 23), (23, 27), (27, 26), (26, 22),
    # Outer Right Goal Line area
    (24, 25), (25, 28), (27, 29),(29,26),

    # --- CENTER LINE & CIRCLE ---
    # Vertical Halfway Line
    (13, 14), (14, 15), (15, 16),
    # Center Circle (Compass Points)
    (14, 31), (31, 15), (15, 30), (30, 14),

    # --- OUTER BOUNDARIES (Sidelines) ---
    # Top Sideline (Right half)
    (13, 24),
    # Bottom Sideline (Right half)
    (16, 29),
    # Top Sideline (Left half)
    (0,13)
]

def draw_keypoints_on_frame(frame, keypoints):
    """Draws black lines and green dots on a single frame"""
    frame_copy = frame.copy()
    
    if keypoints:
        # 1. Draw Lines (Skeleton) - Now in Black
        for start_id, end_id in FIELD_CONNECTIONS:
            if start_id in keypoints and end_id in keypoints:
                pt1 = tuple(map(int, keypoints[start_id]))
                pt2 = tuple(map(int, keypoints[end_id]))
                cv2.line(frame_copy, pt1, pt2, (0, 0, 0), 2)

        # 2. Draw Dots (Keypoints)
        for kp_id, (x, y) in keypoints.items():
            # Green Dot
            cv2.circle(frame_copy, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # Draw ID Labels for confirmation
            cv2.putText(frame_copy, str(kp_id), (int(x)+5, int(y)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 3. Stats
        text = f"Detected: {len(keypoints)} pts"
        cv2.putText(frame_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    else:
        cv2.putText(frame_copy, "Searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return frame_copy

def draw_keypoints_and_lines(frames, keypoints_list):
    """Legacy function - Draws black lines and green dots on the video (loads all frames)"""
    annotated_frames = []
    
    for frame, keypoints in zip(frames, keypoints_list):
        annotated_frames.append(draw_keypoints_on_frame(frame, keypoints))
    
    return annotated_frames

def main():
    print("🚀 Starting Keypoints Test...")
    
    # --- CONFIGURATION ---
    KP_MODEL_PATH = 'models/Keypoints_detection_best.pt'
    INPUT_VIDEO = 'input_videos/match.mp4'
    OUTPUT_VIDEO = 'output_videos/Test.mp4'
    BATCH_SIZE = 10  # Process 10 frames at a time
    
    if not os.path.exists(KP_MODEL_PATH):
        print(f"❌ Error: Model not found at {KP_MODEL_PATH}")
        return
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Error: Video not found at {INPUT_VIDEO}")
        return
        
    os.makedirs('output_videos', exist_ok=True)

    print(f"📹 Opening video: {INPUT_VIDEO}")
    
    # Use context manager for automatic cleanup
    with VideoProcessor(INPUT_VIDEO) as video:
        total_frames = video.total_frames
        print(f"   Total frames: {total_frames}")
        print(f"   Resolution: {video.width}x{video.height}")
        print(f"   FPS: {video.fps}")
        
        print(f"🤖 Loading Model...")
        kp_tracker = KeypointsTracker(
            model_path=KP_MODEL_PATH, 
            conf=0.3, 
            kp_conf=0.5 
        )
        
        # Initialize video writer
        writer = VideoWriter(OUTPUT_VIDEO, video.width, video.height, video.fps)
        
        print(f"🔍 Detecting Keypoints (Batch Size: {BATCH_SIZE})...")
        print("💾 Writing output frames as we process...")
        
        frame_count = 0
        
        # Process frames in batches
        for batch_idx, batch_frames in video.get_batch_generator(batch_size=BATCH_SIZE):
            # Detect keypoints in batch
            kp_detections = kp_tracker.detect(batch_frames)
            
            # Process each frame in the batch
            for frame, detection in zip(batch_frames, kp_detections):
                # Track keypoints
                keypoints = kp_tracker.track(detection)
                
                # Draw on frame
                annotated_frame = draw_keypoints_on_frame(frame, keypoints)
                
                # Write immediately (don't store)
                writer.write(annotated_frame)
                
                frame_count += 1
            
            # Progress update
            if frame_count % 50 == 0 or frame_count == total_frames:
                print(f"   Processed {frame_count}/{total_frames} frames...")
        
        writer.release()
    
    print(f"✅ Done! Output saved to: {OUTPUT_VIDEO}")
    print("💡 Memory-efficient processing: Only processes a few frames at a time!")

if __name__ == '__main__':
    main()
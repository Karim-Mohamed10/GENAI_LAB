from tracker.keypoints_tracker import KeypointsTracker
import cv2
import os

# ==============================================================================
# FIELD CONNECTION MAP (to be corrected based on detected IDs)
# ==============================================================================
FIELD_CONNECTIONS = [
    # --- LEFT SIDE ---
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),   # Left Goal Line
    (6, 7), (7, 8), (8, 9), (9, 6),           # Left Penalty Box (18-yd)
    (10, 12),                                 # Left 6-yard line (11 is isolated spot)

    # --- RIGHT SIDE ---
    (25, 26), (26, 27), (27, 28), (28, 29), (30, 31), # Right Goal Line
    (21, 22), (22, 23), (23, 24), (24, 21),           # Right Penalty Box (18-yd)
    (18, 19),                                         # Right 6-yard line (20 is isolated spot)

    # --- SIDELINES ---
    (0, 13), (13, 25),   # Top Sideline
    (5, 17), (17, 31),   # Bottom Sideline

    # --- CENTER ---
    (13, 14), (14, 15), (15, 16), (16, 17),           # Halfway Line
    (14, 15), (15, 16)                                # Center Circle
]

def draw_keypoints_on_image(image, keypoints):
    """Draws keypoints with IDs and connection lines on image"""
    img_copy = image.copy()
    
    if keypoints:
        # 1. Draw Lines (Skeleton)
        for start_id, end_id in FIELD_CONNECTIONS:
            if start_id in keypoints and end_id in keypoints:
                pt1 = tuple(map(int, keypoints[start_id]))
                pt2 = tuple(map(int, keypoints[end_id]))
                cv2.line(img_copy, pt1, pt2, (0, 0, 0), 2)

        # 2. Draw Keypoints with IDs
        for kp_id, (x, y) in keypoints.items():
            # Green Dot
            cv2.circle(img_copy, (int(x), int(y)), 8, (0, 255, 0), -1)
            
            # ID Label (Yellow text with black outline for better visibility)
            text = str(kp_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Black outline
            cv2.putText(img_copy, text, (int(x)+10, int(y)-10), 
                       font, font_scale, (0, 0, 0), thickness + 2)
            # Yellow text
            cv2.putText(img_copy, text, (int(x)+10, int(y)-10), 
                       font, font_scale, (0, 255, 255), thickness)

        # 3. Stats
        text = f"Detected: {len(keypoints)} keypoints"
        cv2.putText(img_copy, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
    else:
        cv2.putText(img_copy, "No keypoints detected!", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    return img_copy

def main():
    print("🚀 Testing Keypoints on Image...")
    
    # --- CONFIGURATION ---
    KP_MODEL_PATH = 'models/Keypoints_detection_best.pt'
    INPUT_IMAGE = 'input_videos/a9f16c_9_9_png.rf.808c80144837aa0c6e13ccc5f9aec71b.jpg'
    OUTPUT_IMAGE = 'output_videos/keypoints_detected.png'
    
    if not os.path.exists(KP_MODEL_PATH):
        print(f"❌ Error: Model not found at {KP_MODEL_PATH}")
        return
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Error: Image not found at {INPUT_IMAGE}")
        print("   Make sure 'field_2d_v2.png' is in the project root directory")
        return
        
    os.makedirs('output_videos', exist_ok=True)

    print(f"📷 Loading image: {INPUT_IMAGE}")
    image = cv2.imread(INPUT_IMAGE)
    
    if image is None:
        print(f"❌ Error: Could not read image {INPUT_IMAGE}")
        return
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]}")

    print(f"🤖 Loading Model...")
    kp_tracker = KeypointsTracker(
        model_path=KP_MODEL_PATH, 
        conf=0.3, 
        kp_conf=0.5 
    )

    print(f"🔍 Detecting Keypoints...")
    detection = kp_tracker.detect([image])[0]
    keypoints = kp_tracker.track(detection)
    
    print(f"✅ Detected {len(keypoints)} keypoints")
    print("\nKeypoint IDs found:")
    for kp_id in sorted(keypoints.keys()):
        x, y = keypoints[kp_id]
        print(f"   ID {kp_id:2d}: ({x:.1f}, {y:.1f})")

    print("\n🎨 Drawing Keypoints and Lines...")
    annotated_image = draw_keypoints_on_image(image, keypoints)
    
    print(f"💾 Saving annotated image to: {OUTPUT_IMAGE}")
    cv2.imwrite(OUTPUT_IMAGE, annotated_image)
    
    print(f"\n✅ Done! Check {OUTPUT_IMAGE} to see all keypoint IDs")
    print("   Use these IDs to correct the FIELD_CONNECTIONS map")

if __name__ == '__main__':
    main()

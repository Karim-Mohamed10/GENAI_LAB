import cv2
import numpy as np
import json
import os

# --- Configuration ---
POSITIONS_FILE = 'exports/positions.json'
FIELD_IMAGE_PATH = 'input_videos/field_2d_v2.png'

# The exact dimensions of your pitch coordinate system in meters
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0

def generate_team_heatmap(team_data, field_img, team_name):
    h, w = field_img.shape[:2]
    
    # 1. Create a blank grayscale image (density map)
    density_map = np.zeros((h, w), dtype=np.float32)
    
    # 2. Add a tiny "splat" for every footstep a player took
    for pos in team_data:
        x_m, y_m = pos
        
        # Convert meters to pixels
        px = int((x_m / FIELD_LENGTH_M) * w)
        py = int((y_m / FIELD_WIDTH_M) * h)
        
        # Ensure it's inside the pitch bounds
        if 0 <= px < w and 0 <= py < h:
            density_map[py, px] += 1.0  # Add "heat" to this pixel
            
    # 3. Apply a massive Gaussian Blur to turn the dots into "clouds"
    # (Increase these numbers if the clouds are too small, decrease if too blurry)
    density_map = cv2.GaussianBlur(density_map, (0, 0), sigmaX=35, sigmaY=35)
    
    # 4. Normalize the math to a 0-255 scale so OpenCV can color it
    density_map = cv2.normalize(density_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 5. Apply the "Jet" Colormap (Blue = Low, Green = Med, Yellow = High, Red = Very High)
    colored_heatmap = cv2.applyColorMap(density_map, cv2.COLORMAP_JET)
    
    # 6. Create an Alpha Mask (Transparency)
    # We want areas with 0 heat to be fully invisible, and high heat to be ~70% visible
    alpha = (density_map / 255.0) * 0.7 
    alpha = np.expand_dims(alpha, axis=2) # Expand for math (H, W, 1)
    
    # 7. Blend the heatmap cleanly over the actual grass photo
    blended_output = (colored_heatmap * alpha + field_img * (1 - alpha)).astype(np.uint8)
    
    # 8. Save the final image
    output_filename = f"exports/heatmap_Team_{team_name}.png"
    cv2.imwrite(output_filename, blended_output)
    print(f"Saved {output_filename}")

def main():
    if not os.path.exists(POSITIONS_FILE):
        print(f"Error: {POSITIONS_FILE} not found. Run main_full_tracking.py first!")
        return
        
    if not os.path.exists(FIELD_IMAGE_PATH):
        print(f"Error: {FIELD_IMAGE_PATH} not found!")
        return

    print("Loading data...")
    with open(POSITIONS_FILE, 'r') as f:
        data = json.load(f)
        
    field_img = cv2.imread(FIELD_IMAGE_PATH)
    
    # Generate for Team 1
    if "1" in data and len(data["1"]) > 0:
        print("Generating Heatmap for Team 1...")
        generate_team_heatmap(data["1"], field_img.copy(), "1")
        
    # Generate for Team 2
    if "2" in data and len(data["2"]) > 0:
        print("Generating Heatmap for Team 2...")
        generate_team_heatmap(data["2"], field_img.copy(), "2")

    print("Done! Check your folder for the new heatmap images.")

if __name__ == '__main__':
    main()
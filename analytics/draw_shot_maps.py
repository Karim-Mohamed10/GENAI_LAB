import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    # Load shot data
    try:
        with open('exports/shots.json', 'r') as f:
            shots = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find exports/shots.json")
        return

    # Identify unique teams
    teams = set(s.get("shooter_team") for s in shots if "shooter_team" in s and s.get("shooter_team") is not None)

    # Setup pitch image
    img_path = 'input_videos/field_2d_v2.png'
    try:
        img = mpimg.imread(img_path)
    except FileNotFoundError:
        print(f"Error: Could not find pitch image at {img_path}")
        return

    # Plot shot map for each team
    for team in teams:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        # Display the background pitch
        ax.imshow(img, extent=[0, 105, 68, 0])
        
        for s in shots:
            if s.get("shooter_team") == team:
                start_pos = s.get("start_pos")
                end_pos = s.get("end_pos")
                outcome = s.get("outcome", "")
                
                if not start_pos or not end_pos:
                    continue
                    
                x1, y1 = start_pos
                x2, y2 = end_pos
                
                if "ON_TARGET" in outcome:
                    # Draw a blue line for on-target shots
                    ax.plot([x1, x2], [y1, y2], color="blue", linewidth=1.5)
                    # White circle with black edge at start
                    ax.plot(x1, y1, marker='o', markerfacecolor="white", markeredgecolor="black", markersize=6)
                    # Blue circle with black edge at end
                    ax.plot(x2, y2, marker='o', markerfacecolor="blue", markeredgecolor="black", markersize=6)
                
                else: # OFF_TARGET / BLOCKED
                    # Draw a red dashed line for missed/blocked shots
                    ax.plot([x1, x2], [y1, y2], color="red", linestyle="--", linewidth=1.5)
                    # White circle with black edge at start
                    ax.plot(x1, y1, marker='o', markerfacecolor="white", markeredgecolor="black", markersize=6)
                    # Large red 'X' marker at end
                    ax.plot(x2, y2, marker='X', color="red", markersize=10)
        
        # Turn off axes and set title
        ax.axis('off')
        ax.set_title(f"Shot Map - Team {team}")
        
        # Save figure
        output_filename = f"exports/shot_map_team_{team}.png"
        fig.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    main()
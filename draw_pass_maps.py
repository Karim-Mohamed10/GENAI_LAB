import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    # Load pass data
    try:
        with open('passes.json', 'r') as f:
            passes = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find passes.json")
        return

    # Identify unique teams
    teams = set(p.get("initiator_team") for p in passes if "initiator_team" in p)

    # Setup pitch image
    img_path = 'input_videos/field_2d_v2.png'
    try:
        img = mpimg.imread(img_path)
    except FileNotFoundError:
        print(f"Error: Could not find pitch image at {img_path}")
        return

    # Plot pass map for each team
    for team in teams:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        
        # Display the background pitch
        ax.imshow(img, extent=[0, 105, 68, 0])
        
        for p in passes:
            if p.get("initiator_team") == team:
                start_pos = p.get("start_pos")
                end_pos = p.get("end_pos")
                status = p.get("status")
                
                if not start_pos or not end_pos:
                    continue
                    
                x1, y1 = start_pos
                x2, y2 = end_pos
                
                if status == "COMPLETED":
                    # Draw a blue line
                    ax.plot([x1, x2], [y1, y2], color="blue", linewidth=1.5)
                    # White circle with black edge at start
                    ax.plot(x1, y1, marker='o', markerfacecolor="white", markeredgecolor="black", markersize=6)
                    # White circle with black edge at end
                    ax.plot(x2, y2, marker='o', markerfacecolor="white", markeredgecolor="black", markersize=6)
                
                elif status == "INTERCEPTED":
                    # Draw a red dashed line
                    ax.plot([x1, x2], [y1, y2], color="red", linestyle="--", linewidth=1.5)
                    # White circle with black edge at start
                    ax.plot(x1, y1, marker='o', markerfacecolor="white", markeredgecolor="black", markersize=6)
                    # Large red 'X' marker at end
                    ax.plot(x2, y2, marker='X', color="red", markersize=10)
        
        # Turn off axes and set title
        ax.axis('off')
        ax.set_title(f"Pass Map - Team {team}")
        
        # Save figure
        output_filename = f"pass_map_team_{team}.png"
        fig.savefig(output_filename, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Successfully generated {output_filename}")

if __name__ == "__main__":
    main()

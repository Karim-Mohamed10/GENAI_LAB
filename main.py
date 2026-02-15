from utils.video import read_video,save_video
from tracker.tracker import Tracker
def main():
    
    video_frames=read_video('input_videos/match.mp4')
    
    tracker=Tracker('models/best.pt')
    
    track=tracker.get_object_tracks(video_frames)
    
    annotaed=tracker.draw_annotations(video_frames,track)
    
    save_video('output_videos/output.mp4',annotaed)

if __name__=='__main__':
    main()
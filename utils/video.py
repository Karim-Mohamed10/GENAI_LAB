import cv2
import numpy as np

def read_video(video_path):
    """Legacy function - loads all frames into memory. Use VideoProcessor for large videos."""
    vid=cv2.VideoCapture(video_path)
    frames=[]
    while True:
        state,frame=vid.read()
        if not state:
            break
        frames.append(frame)
    return frames

def save_video(video_path, video_frames):
    if len(video_frames) == 0:
        return
    height, width, _ = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    for frame in video_frames:
        out.write(frame)
    
    out.release()


class VideoProcessor:
    """Memory-efficient video processor that yields frames instead of loading all at once."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
    
    def get_batch_generator(self, batch_size=1):
        """
        Generator that yields batches of frames.
        
        Args:
            batch_size: Number of frames per batch
            
        Yields:
            tuple: (batch_index, frames) where frames is a list of numpy arrays
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        batch_idx = 0
        while True:
            batch = []
            for _ in range(batch_size):
                ret, frame = self.cap.read()
                if not ret:
                    if batch:  # Yield remaining frames
                        yield batch_idx, batch
                    return
                batch.append(frame)
            
            yield batch_idx, batch
            batch_idx += 1
    
    def reset(self):
        """Reset video to beginning."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


class VideoWriter:
    """Wrapper for cv2.VideoWriter with easier initialization."""
    
    def __init__(self, output_path, width, height, fps=30):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")
    
    def write(self, frame):
        """Write a single frame."""
        self.writer.write(frame)
    
    def write_batch(self, frames):
        """Write multiple frames."""
        for frame in frames:
            self.writer.write(frame)
    
    def release(self):
        """Release the video writer."""
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

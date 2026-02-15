from ultralytics import YOLO

model=YOLO('models/best.pt')

results=model.predict('input videos/match.mp4',save=True)
from ultralytics import YOLO

model = YOLO("yolo11l.pt")

#result = model.predict("input_videos/video.mp4", save =True)

result = model.track("input_videos/video.mp4", save =True, persist=True)
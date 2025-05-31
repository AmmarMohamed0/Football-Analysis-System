import cv2

# Create a read video Function 
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames=[]

    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        frames.append(frame)
    return frames

# Create a video save Function 
def save_video(output_video_frames, output_video_path, fps=24):
    if not output_video_frames:
        raise ValueError("Frame list is empty")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))

    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
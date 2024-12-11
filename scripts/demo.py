from ultralytics import YOLO
import cv2

model = YOLO(model = "./models/yolov11m-best.pt")

cap = cv2.VideoCapture("./video/demo.mkv")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("./video/demo-labeled.avi", fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame)

    # Get the annotated frame
    annotated_frame = results[0].plot()

    # Write the frame to the output video
    out.write(annotated_frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()

# takes about 250ms per frame on my computer
# https://www.youtube.com/watch?v=IjlaQoe_y68
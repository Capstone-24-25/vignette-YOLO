# This script demonstrates how to replicate our results (training YOLO 11m)
import yolo_preprocess # preprocess data

from ultralytics import YOLO # import ultralytics

model = YOLO("yolo11m.pt") # specify model to train

trained_model = model.train(data = "./data/ultralytics/data.yaml", epochs=50, imgsz=640, batch=-1) # specify hyperparameters and train model
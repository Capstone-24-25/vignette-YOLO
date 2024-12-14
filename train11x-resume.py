# This scripts is used to train the YOLO11x model on CSC resources
# conda environment must be set up with the following commands (run on supercomputer), these commands are for use on pod
# 1. $ source /sw/csc/anaconda/anaconda3/bin/activate
# 2. $ conda create --name ultralytics-env python=3.11 -y
# 3. $ conda activate ultralytics-env
# 4. $ conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics


from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")

results = model.train(resume = True, epochs=300, imgsz=640, batch=16, cache=True, device = [0, 1, 2, 3])

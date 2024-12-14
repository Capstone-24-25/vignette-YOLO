# Military Aircraft Detection Using YOLO ✈️

A vignette showcasing the development of a YOLO (You Only Look Once) object detection model designed to identify, classify, and localize various types of military aircraft, created as part of a class project for PSTAT197A in Fall 2024.

## Countributors 👥

Bennett Bishop, Joseph Zaki, Luke Dillon, Benjamin Drabeck,	Samantha Su

## Vignette Abstract ✍️

Vignette topic: This project involves building a YOLO (You Only Look Once) object detection model to identify, classify, and localize different types of military aircraft. The model is trained on the Military Aircraft Detection Dataset, which contains images and annotations of various military aircraft types. Our goal is to develop a robust and efficient object detection model capable of detecting multiple aircraft types in an image with high accuracy, while leveraging the YOLO architecture for real-time performance.

Example Data: The project uses a curated dataset of military aircraft containing images and bounding box annotations for various 74 aircraft types such as Tornado, A400M, F18, etc.

Data Source: [Military Aircraft Detection](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)

This project utilizes YOLOv11, specifically the YOLOv11n implementation by Ultralytics, to perform object detection tasks. The model, with 2.6 million parameters and 218 layers, achieved a mean Average Precision (mAP) of 0.567 at a 0.5 Intersection over Union (IoU) threshold. While the model performs moderately well, the results suggest that a larger model could yield better performance.

## Repository Contents 📙

- data: Contains the dataset files for the project.
  - consolidated.csv – a combined dataset used for analysis or training.
- models: Stores machine learning models.
  - yolo11x-50epochs-results – Contains results.csv and images from yolo11x model training (model is too large to upload to GitHub)
  - yolov11m-best.pt – A trained YOLO model (medium version).
  - yolov11n-best.pt – A trained YOLO model (nano version).
- scripts: Includes Python scripts for data processing and preprocessing.
  - slurm/ – Contains job files for training on [UCSB CSC Pod Cluster](https://csc.cnsi.ucsb.edu/)
  - training-hpc – Contains python files used for training (or continuing training) yolo11x model on [UCSB CSC Pod Cluster](https://csc.cnsi.ucsb.edu/)
  - consolidatecsvs.py – script to merge multiple CSVs into a consolidated dataset.
  - demo.py – Script to demonstrate the YOLO model's performance on videos or images.
  - sample_yolo.py – Script showcasing how to use the YOLO model on sample data.
  - yolo_preprocess.py – Script to prepare data for the YOLO model.
- video: Contains video files for demonstration purposes.
  - demo-labeled.avi – A video with labeled predictions from the YOLO model.
  - demo.mkv – A raw input video used for testing or showcasing the model.
- .DS_Store: A macOS system file that can be ignored.
- .gitignore: Specifies files and directories to exclude from version control.
- README.md: Provides an overview of the project
- yolo11-inference.ipynb: Jupyter Notebook for running the main analysis or workflows. It serves as an interactive guide of our methods by integrating code with step-by-step explanations.

## Reference List 🧑‍🎓

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016, June). You Only Look Once: Unified, Real-Time Object Detection. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Jiang, P., Ergu, D., Liu, F., Cai, Y., & Ma, B. (2022). A Review of Yolo Algorithm Developments. Procedia Computer Science, 199, 1066–1073. doi:10.1016/j.procs.2022.01.135

## Instructions on Contributing 🤔

1. Fork the Repository
2. Clone the Forked Repository to your local machine

```
git clone https://github.com/Capstone-24-25/vignette-YOLO.git
cd vignette-YOLO
```

3. Create a new branch for your contribution

```
git checkout -b <branch-name>
```

4. Make your changes and ensure the project still works as expected. Use pytest to run tests if provided
5. Push Changes to Your Fork
6. Open a Pull Request

##  Instructions on Use 📝

We trained two models stored in the models directory, yolov11n-best and yolov11m-best. Please use the yolov11m-best model, as it is our better performing model.

To load and use this model use the ultralytics package and follow the following steps.
- run the yolo_preprocess.py script in the scripts folder
- import YOLO from ultraytics
- find the **absolute** path (not relative) to the model directory (e.g. mine is: C:\Users\luke\Documents\class\pstat197a\vignette-YOLO\models\yolov11m-best.pt)
- run this code to load the model: model = YOLO("C:\\Users\\luke\\Documents\class\\pstat197a\\vignette-YOLO\\models\\yolov11m-best.pt", task="detect")

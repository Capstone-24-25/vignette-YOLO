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
  - yolov11n-best.pt – the trained YOLO model file.
- scripts: Includes Python scripts for data processing and preprocessing.
  - consolidatecsvs.py – script to merge multiple CSVs into a consolidated dataset.
  - yolo_preprocess.py – script to prepare data for the YOLO model.
- .gitignore: Specifies files and directories to exclude from version control.
- README.md: Provides an overview of the project
- main.ipynb: Jupyter notebook for running the main analysis or workflows.

## Reference List 🧑‍🎓

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016, June). You Only Look Once: Unified, Real-Time Object Detection. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Jiang, P., Ergu, D., Liu, F., Cai, Y., & Ma, B. (2022). A Review of Yolo Algorithm Developments. Procedia Computer Science, 199, 1066–1073. doi:10.1016/j.procs.2022.01.135

## Instructions on Contributing

1. Fork the Repository
2. Clone the Forked Repository to your local machine

```
git clone https://github.com/Capstone-24-25/vignette-YOLO.git
cd vignette-YOLO
```

3. Create a new branch for your contribution
4. Make your changes and ensure the project still works as expected. Use pytest to run tests if provided
5. Push Changes to Your Fork
6. Open a Pull Request

##  Instructions on Use (draft)



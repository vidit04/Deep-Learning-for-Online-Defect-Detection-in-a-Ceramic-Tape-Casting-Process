# Deep-Learning-for-Online-Defect-Detection-in-a-Ceramic-Tape-Casting-Process
### Contents of Thesis
1. Motivation
2. Aims and Objectives
3. Methodology
4. Results 
5. Conclusion

## 1. Motivation

## 2. Aims & Objectives

To develop a Neural Network algorithm for pattern recognition in ceramic tape reflected mode images based on category of defect. Category of defect in ceramic tape images includes
* No Defect
* Small Defect
* Strips
* Streaks
* Surface Irregularities
* Technology Defect

Sample images for category of defects in ceramic tape images are shown below.

![](images/cat.png)

### Objectives

The aim of the thesis is achieved by the following objectives.

1. Collecting high-resolution ceramic tape image data of 2 x 8192 pixels dimensions and slicing it into smaller image dimensions of 500 x 500 pixels in Python.
2. Converting the ceramic tape image data into a labeled dataset.
3. Creating training routine for different neural network architectures in Python using Keras and TensorFlow.
4. Training different neural network architectures on labeled ceramic tape dataset for binary classification.
5. Selecting better performing model from binary classification results and training them on ceramic tape dataset for multi-classification.
6. Evaluation of architecture performance for binary classification and multi-classification.

## 3. Methodology
#### 3.1 Data Collection
Raw image data collected from line camera is of dimension 2 x 8192 pixels. High resolution image data include images of moving ceramic tape in reflected mode and transmission mode. In a single experiment of ceramic tape casting, 50 m (approx.) of ceramic tape is produced and 300,000 (approx.) high resolution images are collected. High resolution reflected mode images of dimension 8192 x 8192 are created by stitching reflected mode array and transmission mode array from the raw source image in sequence. Large reflected mode images of dimension 8192 x 8192 pixels are sliced into smaller 500 x 500 pixels images as shown in the figure below.
![](images/col.png)



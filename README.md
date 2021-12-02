# IPFImage

This repository is created for the paper:
Deep Learning-based radiomic features predicts decline of Forced Vital Capacity in adults with Idiopathic Pulmonary Fibrosis 
Purpose
Idiopathic pulmonary fibrosis (IPF) is a progressive, life-threatening, interstitial lung disease of unknown etiology [1]. IPF is characterized by progressive interstitial fibrosis caused by the thickening and scarring of the lung [1]. The average life expectancy of an IPF patient after diagnosis is 3-4 years. Unfortunately, the speed of disease progression has high variance and doctors are currently not able to easily tell if an IPF patient’s lungs will achieve long-term stability or rapidly deteriorate [1]. Currently forced vital capacity (FVC) is the best indicator of IPF disease progression [2, 3, 4]. FVC is a lung function test which measures the amount of air that can be forcibly exhaled from your lungs after taking the deepest breath possible. We are mainly interested in visual markers of IPF and will study the efficacy of various visual features. 

![image]{ width: 200px; }(https://user-images.githubusercontent.com/49659087/144462701-b21abfbc-3149-4f9d-9c07-b76da070a388.png)
Figure 1 fibrosis(left) vs healthy(right) lung. Red circled regions are so called honeycombing area. 

## Step 1. Segmentation of images: 
Segmentation.ipynb
![image](https://user-images.githubusercontent.com/49659087/144463140-850899ac-12fe-4bce-9d7a-d36840ef313e.png)
Since the region of interest for us is only the lung area, we perform a segmentation on the image to remove the rib cage and fat tissues. A windowing technique is applied with a uniform clip from - 1200 to 200 Hounsfields (HU) to remove differences between scans (figure 2 middle). We then used a pre-trained U-net model [16] to segment the lung area (figure 2 right).

## Step 2. extract texture features using GLCM
TextureFeature-single-analysis-glcm.ipynb

## Step 3 extract features using object detection with a pretrained CNN

3.1 CNN training with ILD dataset
OD-CNN-pipeline-efn-cv5.ipynb
3.2 perfrom object detection 
ObjectDetection-Pipeline.ipynb
3.3 visualize objection results
ObjectDetection-visual.ipynb

## Step 4. Benchmark
Benchmark feature.ipynb

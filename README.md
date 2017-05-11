
# Vehicle Detection Project

[//]: # (Image References)
[training_data]: ./output_images/training_data.png
[bin_spatial]: ./output_images/bin_spatial.png
[color_hist]: ./output_images/color_hist.png
[features_HOG]:  ./output_images/features_HOG.png
[features_extraction]: ./output_images/features_extraction.png
[search_area]: ./output_images/search_area.png
[image_labeled]: ./output_images/image_labeled.png
[pipline]: ./output_images/pipline.png
[video1]: ./project_video.mp4

#### The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.



---
## 1. Data Exploration

Labeled images were taken from the Udacity provided images [Vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [Non-Vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) . All images are 64x64x3 pixels. In total there are 8792 images of vehicles and 8968 images of non vehicles. Thus the data is slightly unbalanced with about 2% more non vehicle images than vehicle images Shown below is an example of each class (vehicle, non-vehicle) of the data set. The data set is explored in the notebook Vehicle_Detection.ipynb see Chapter 1 "Trainingdata"

![alt text][training_data]

---

## 2. Feature Extraction 

In notebook chapter 2 "Feature extrection" I have defined different feature extraction methods.

 * Colour-space conversion. Allows for conversion to one of the OpenCV supported colour spaces like RGB, HSV, LUV, HLS, YUV, YCrCb 

 * Extract spatial features. Extract features by collecting colour spatial information (per channel) and concatenating to form a feature vector. 
 
 ![alt text][bin_spatial]

* Extract colour histograms. Using `np.histogram` we compute histograms for each image channel and then concatenate them together to form the features. See notebook Chapter 2.1 "Spatial color features"

 ![alt text][color_hist]

* Extract HOG (Histograms of Oriented Gradients) features. The `skimage.feature.hog` function is used here to extract features per image channel. See notebook Chapter 2.3. "HOG features" Here is an example with following HOG parameters `orient=8, pix_per_cell=8, cell_per_block=2`

 ![alt text][features_HOG]
    
### 2.1 Extract Feature Combination

In the notebook Chapter 4  "Extract Feature Combinations" the method `extract_features` is implemented. It combines the different features. 
Here is an example of a combination of spatial, color histogram and HOG features with following paramaeters 
`color_space = 'YCrCb', orient = 9, pix_per_cell = 8  cell_per_block = 2, hog_channel = 'ALL',  spatial_size = (16, 16), hist_bins = 16`

![alt text][features_extraction]

---


## 3. Training a classifier

First I extracted the features from the Udacity training sets of vehicles and non-vehicles and created a feature and label set (notebook Chapter 4 "Extract Feature Combinations" Cell 2) and normalized the features with a `sklearn.preprocessing.StandardScaler`. (notebook Chapter 4 "Extract Feature Combinations" Cell 3) , which scaled the data to zero mean and unit variance.  For the training I splitted th e test set into a test (80%) and a validation set (20%) of the labeled data input. 

Than I try to find a SVC via `RandomizedSearchCV {'kernel':('linear', 'rbf'), 'C':[1, 5]}` and `GridSearchCV{'kernel':('linear', 'rbf'), 'C':[1, 5]}`}). 
Aditionally I choosed `LinearSVC ( {'C':[1, 5]} )` 

I was able to consistently achieve accuracy of about 98-99,9%.


| Kernel | C | Orient | Pixels/Cell | Cell/ Block |Spatial|Histogram|Color| Accuracy | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|rbf | 5 | 9 | 8 | 2 |X|X| YCrCb| 0.993
| linear | 5 | 9 | 8 | 2 |X|X|  YCrCb | 0.990
| linear | 1 | 9 | 8 | 2 |X|X|  YCrCb | 0.991
| rbf | 5 | 9 | 8 | 2 ||X|  YCrCb| 0.9932
| linear | 5 | 9 | 8 | 2 ||X|  YCrCb| 0.986
| linear | 1 | 9 | 8 | 2 ||X|  YCrCb| 0.987
| linear | 1 | 9 | 8 | 2 |X|X|  YUV | 0.991
| linear | 5 | 9 | 8 | 1 |X|X|  YCrCb | 0.993
| linear | 1 | 9 | 8 | 1 |X|X|  YCrCb | 0.991
|rbf | 5| 9 | 8 | 1 |X|X|  YCrCb| 0.997

---


## 4. Sliding Window Search

In the notebook Chapter 4  "Sliding Window" I adapted the method find_cars from the lesson materials. 

Here you can see search areas by the window sizes and positions overlaps.

![alt text][search_area]

The method combines HOG feature extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier. The method performs the classifier prediction on the HOG features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.

![alt text][image_labeled]

---

## 5.  Filter for false positives - Heatmap and Threshold and Labeling

Because a true positive is typically accompanied by several positive detections (see Boxed), while false positives are typically accompanied by only one or two detections, a combined heatmap and threshold is used to differentiate the two. The  `add_heat`(see notebook Chapter 5.1 'Heatmap') function increments the pixel value of an all-black image the size of the original image at the location of each detection rectangle. Areas encompassed by more overlapping rectangles are assigned higher levels of heat. 
The `apply_threshold`function sets all pixels that don't exceed the threshold (typical 1-2 ) to zero. 
The `scipy.ndimage.measurements.label()` function collects spatially contiguous areas of the heatmap and assigns each a label.

The following image shows the pipline on the test images:

![alt text][pipline]

---

## 6. Video Implementation

### 1.  Prozess Pipline

I choosed following prozess pipeline (see notebook chapter 6.1 "Prozess pipeline")

* Extract Features
* Train SVC
* Find Cars via SVC and Sliding Window Search
* Filter for false positives via heatmap and thresholding
* Label the findings and draw boxes

### 2. Select Final Training Parameter 

| Kernel | C | Orient | Pixels/Cell | Cell/ Block | Color|Spatial|Histogram| False Positive | False Negative | Time per Frame
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|rbf | 5 | 9 | 8 | 2 |X|X| YCrCb| 7 < 1s  | < 1s | 9.48 s
| linear | 5 | 9 | 8 | 2 |X|X|  YCrCb |22 < 1s| 1-2 s | 0.81 s 
| linear | 1 | 9 | 8 | 2 |X|X|  YCrCb | 24 < 1s| 1-2 s | 0.85 s
|rbf | 5 | 9 | 8 | 2 ||X| YCrCb| 5 < 1s | < 1 s | 8.27 s
| linear | 5 | 9 | 8 | 2 ||X|  YCrCb| 36  1-2 s| 1 - 2 s | 0.77
| linear | 1 | 9 | 8 | 2 ||X|  YCrCb| 39 1-2 s| 1 - 2s | 0.77 
| linear | 1 | 9 | 8 | 2 |X|X|  YUV  |  30 1-2 s | 2 - 3 s| 0.70| 0.991
| **linear** | **5** | **9** | **8** | **1** |**X**|**X**|  **YCrCb**  | **6  < 1s**|  **< 1s** | **0.76** | **0.993**
| linear | 1 | 9 | 8 | 1 |X|X|  YCrCb  |15 < 1s| < 1s | 0.76
|**rbf** | **5**| **9** | **8** | **1** |**X**|**X**|  **YCrCb**| **1 <0,5** | **< 0,5** |  **8.69**

The best result of prozessing a frame did the rbf SVC. But the  prozess time per videoframe was by the verry slow (8.69)

The linear SVC with a C=5 did the relativly best result if you see the the prozess time per videoframe. But it detects a lot of false positivs.

---

### 3. Video Prozessing

|  RBF SVC (C=5)                 | Linear SVC (C=5)                        |
|---------------------------|---------------------------|
[![E](https://img.youtube.com/vi/ddSVKLMnd30/0.jpg)](https://youtu.be/ddSVKLMnd30 "RBF SVC (C=5)") | [![E](https://img.youtube.com/vi/q98BRlCPcHw/0.jpg)](https://youtu.be/q98BRlCPcHw "Linear SVC (C=5)")|


---

## 7. Discussion

HOG Features is a very strong technic to to extract shape informations from a image. In the combination with color features it extractes prity unique characteristic of an images.

The classifer is an other importend factor. On one hand it has has to give good enought predictions to find a car on an image. On the otherhand it has to be quick enought to be able to prozzes images in "realtime". So it cold be better to improve the quicker Linear SVC approach and reduse the false positive findings.

To improve the quality of the Linear SVC and reduce the false positiv findings I would try to reduce the area to search in. It could be done by reducing the sliding window search area to the right lane of the street. 

I took some time to find other classifer to do the work of car detection. It could be worth to look an Neuronal Networks like [yolo or  SSD](https://github.com/weiliu89/caffe/tree/ssd) . A NN approach seam to be a lot faster than a SVC approach. An other advantage is, that there is no **serial** seperate feature extraction. An NN like yolo does all in one.
 
 



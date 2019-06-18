# Chest_Xray_normal_pneumonia_classification
Classification of chest X-Ray images into normal and pneumonia

Datasets obtained from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Original source: http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

# Project Aim:
### Binary classification of chest X-Ray images of into 2 classes:
  * `NORMAL`
  * `PNEUMONIA`
  
#### Short overview of the dataset:

  * The data is provided in three datasets: Train, Val and Test. The number of images in each dataset and in each class is shown as below.
  
  <img src= 'https://github.com/xiey1/Chest_Xray_normal_pneumonia_classification/blob/master/images/Data_distribution.png' width=300px>
  
  * The training dataset contains 5216 images, validation dataset 16 images and test dataset 624 images.
  
  * The images are in grayscale (some have 3 channels but can be converted to grayscale).
  
  * The images have different sizes.
    Here is a panel of representative images from each class:
  <img src= 'https://github.com/xiey1/Chest_Xray_normal_pneumonia_classification/blob/master/images/X-Ray_images_view.png' width=500px>

# Challenge:
The biggest challenge of this project is the imbalance of the dataset. 
 * The number of X-Ray images for `NORMAL` and `PNEUMONIA` cases is not 50%/50% in training and test datasets. 
 
 * The class ratio is not consistent across different datasets. The NORMAL/PNEUMONIA ratio is around 1:3 in training dataset, 1:1 in validation dataset, and around 1:1.67 in test dataset.
 
 * The validation dataset is too small and only contains 16 images. This may lead to significant bias as the trained model overall is in favor of both training and validation sets.
 
 * Overall, the dataset is relatively small, and may lead to overfitting and low prediction accuracy on test dataset.
 
 * The 'best' evaluation metric needs to be explored for this project.
 
### Potential solutions:
 * Create a new validation dataset by stratifying the training dataset into training and validaiton sets. The class ratio needs to be kept equal in each dataset to achieve consistent prediction performance scores.
 
 * Add class weight to the training model to avoid potential bias caused by the class imbalance in the dataset.
 
 * In addition to building convolutional neural network (CNN) from stratch, transfer learning can be applied which sometimes could largely increase prediction efficiency for project with small datasets. 
 
 * Multiple evaluation metrics such as overall prediction accuracy, accuracy for each class, precision, recall, F1 score and AUC_ROC (Receiver Operating Characteristic) can be computed.
 
# Approach -- Convolutional Neural Network:
### CNN build from scratch
Here I first tried to build a 14-layer CNN model from scratch:

**((ConV-ReLU)x3-MaxPool)x3 --flattening-- FC1-ReLU-FC2**

![CNN_scratch](https://github.com/xiey1/Chest_Xray_normal_pneumonia_classification/blob/master/images/Image1_CNN.png)

However, the prediction accuracy for `NORMAL` class is really low (35.470%) with very high false positive rate, while the prediction accuracy for `PNEUMONIA` class is very good (98.974%) (the score can also be viewed in the **result** section). I hypothesize the low precision score (0.719) and high recall score (0.990) is due to class imbalance and apply class weight to re-train the model. Although the training performance now improved with `NORMAL` accuracy 51.282%, `PNEUMONIA` accuracy 98.205%, precision 0.771 and recall 0.982, the model is still suboptimal and I decided to try transfer learning. Of note, the overall training/validation accuracy can be more than 90% while the overall accuracy of the test dataset is much lower (80.609% after adding class weight), this suggests overfitting during training process. 

### Transfer Learning with Inception_v3
1. `Inception_v3`: Freeze all the convolutional layers before the final classifer and train the classifier
2. `Inception_v3_w`: Add class weight to `Inception_v3` during training process
3. `Inception_v3_fc2`: Freeze all the convolutional layers, add a fully-connected layer with 256 nodes before the final classifier and train both the fully-connected layer as well as the final classifier
4. `Inception_v3_fc2_w`: Add class weight to `Inception_v3_fc2` during training process

# Result and Evaluation:
The summary of the overall prediction accuracy and accuracy for each class, precision/recall/F1 score, and the results obtained from the confusion matrices (true positive, true negative, false positive, false negative) in test dataset is illustrated here:

<img src= 'https://github.com/xiey1/Chest_Xray_normal_pneumonia_classification/blob/master/images/Prediction_Summary.png' width=900px>

As we can see from the summary metrics, transfer learning using pretrained Inception_v3 model with trained fully-connected layer and classifier (Inception_v3_fc2) achieves the best performance:
   * Overall prediction accuracy: 0.877
   * Precision: 0.872
   * Recall: 0.941
   * F1 score: 0.905
   * ROC_AUC: 0.929

Adding class weight to both Inception_v3 and Inception_v3_fc2 models can increase precision but in sacrifice of recall. In order to achieve a good balance between precision and recall scores, I decide to use the `Inception_v3_fc2` model without adding class weight during training process.

# Discussion:
1. For binary classification problem, it is important to keep an approximate 50%/50% class ratio if possible to avoid the condition in which the trained model is in favor of improving prediction accuracy of the class that carries more weight while ignoring the other class.

2. For situations with imbalanced dataset, we can either decrease the number of data points for the overrepresented class or try to collect more data for the underrepresented class. If neither approach is applicable, we can apply class weight during the training process.

3. It is helpful to compute multiple evaluation metrics for binary classification questions. It can give us a more comprehensive understanding of the prediction performance of the trained model as well as provide direction of future improvement.

4. We should consider the real medical application of this classification model and evaluate the potential risk of false positives and false negatives. It's possible that the automated medical image classification will be combined with other medical tests to help doctors make the final diagnosis. In this situation, it's also important to consider the performance of other tests when we fine-tune the hyperparameters of the model to achieve the best balance between false positive / false negative rate.

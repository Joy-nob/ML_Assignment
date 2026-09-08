# Problem Set 01: CNN-Based Pneumonia Classification

## 1. Problem Description

The objective of this assignment is to develop a Convolutional Neural Network (CNN) to classify pediatric chest X-ray images into two classes:

* **NORMAL**
* **PNEUMONIA**

The dataset contains chest X-ray images divided into training, validation, and test sets.

## 2. Dataset

The chest X-ray dataset was provided externally as part of the assignment and was stored in Google Drive for use with Google Colab.

The dataset contains two classes:

* NORMAL
* PNEUMONIA

The images are grayscale JPEG images with varying original dimensions.

The dataset was not uploaded to GitHub because of its size. The notebook accesses the dataset from Google Drive.

### Dataset Distribution

The original dataset inspection found:

| Split      | NORMAL | PNEUMONIA | Total |
| ---------- | -----: | --------: | ----: |
| Train      |  1,341 |     3,876 | 5,217 |
| Test       |    234 |       390 |   624 |
| Validation |      8 |         8 |    16 |

Because the provided validation set contained only 16 images, a new stratified validation set was created from the training data.

The resulting split was:

| Split      | NORMAL | PNEUMONIA | Total |
| ---------- | -----: | --------: | ----: |
| Training   |  1,207 |     3,488 | 4,695 |
| Validation |    134 |       388 |   522 |
| Test       |    234 |       390 |   624 |

The test set was kept untouched until final evaluation.

## 3. Data Preprocessing

The following preprocessing steps were applied:

1. Images were loaded as grayscale images.
2. Images were resized to **224 × 224** pixels.
3. Pixel values were normalized from the range 0–255 to **0–1**.
4. Training and validation datasets were created using TensorFlow's `tf.data` pipeline.
5. A batch size of **32** was used.
6. The training data was shuffled and prefetched for efficient training.

The images were also checked for corruption, and no corrupted images were found.

## 4. Data Augmentation

To improve generalization and reduce overfitting, augmentation was applied during training.

The following transformations were used:

* Random horizontal flipping
* Random rotation of up to approximately 5%
* Random zoom of up to approximately 5%

Augmentation was applied only during training. The validation and test images were not augmented.

## 5. Handling Class Imbalance

The training data contained considerably more pneumonia images than normal images.

Class weights were therefore used during training:

* **NORMAL:** 1.9449
* **PNEUMONIA:** 0.6730

This gives greater importance to the underrepresented NORMAL class during loss calculation.

## 6. CNN Architecture

A custom CNN was developed using TensorFlow/Keras.

The architecture consists of:

```text
Input: 224 × 224 × 1
        ↓
Data Augmentation
        ↓
Conv2D (32 filters, 3×3)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filters, 3×3)
        ↓
MaxPooling2D
        ↓
Conv2D (128 filters, 3×3)
        ↓
MaxPooling2D
        ↓
GlobalAveragePooling2D
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.5)
        ↓
Dense (1, Sigmoid)
```

The model was compiled using:

* **Optimizer:** Adam
* **Learning rate:** 0.001
* **Loss function:** Binary Cross-Entropy
* **Batch size:** 32
* **Maximum epochs:** 15
* **Early stopping:** Enabled with validation loss monitoring

The best validation weights were restored after training.

## 7. Training Results

Training was performed using a GPU in Google Colab.

The model trained for 11 epochs before early stopping.

The best observed validation performance was approximately:

* **Validation Accuracy:** 78.35%
* **Validation Loss:** 0.4388

Training and validation accuracy/loss plots are included in the notebook.

## 8. Test Results

The trained model was evaluated on the untouched test set containing 624 images.

### Overall Performance

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **62.34%** |
| Precision | **75.58%** |
| Recall    | **58.72%** |
| F1-score  | **66.09%** |

### Classification Report

| Class                | Precision | Recall |   F1-score | Support |
| -------------------- | --------: | -----: | ---------: | ------: |
| NORMAL               |    49.84% | 68.38% |     57.66% |     234 |
| PNEUMONIA            |    75.58% | 58.72% |     66.09% |     390 |
| **Overall Accuracy** |           |        | **62.34%** | **624** |

## 9. Confusion Matrix

The confusion matrix obtained on the test set was:

```text
                 Predicted
              NORMAL  PNEUMONIA
Actual NORMAL    160       74
       PNEUMONIA 161      229
```

The model correctly classified 160 NORMAL images and 229 PNEUMONIA images.

## 10. Findings

The CNN successfully learned patterns from the chest X-ray training data and achieved approximately 62.34% accuracy on the unseen test set.

The results show that the model performed better at precision for the PNEUMONIA class, while the NORMAL class had higher recall. The difference between validation and test performance also indicates that the model did not generalize perfectly to the unseen test images.

The experiment demonstrates the complete CNN classification workflow, including data preprocessing, augmentation, class imbalance handling, model training, and evaluation.

## 11. Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Pillow
* Google Colab
* Google Drive


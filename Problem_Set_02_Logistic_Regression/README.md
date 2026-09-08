# Problem Set 02 — Bank Marketing Logistic Regression

## 1. Overview

This project develops a **Logistic Regression classifier** to predict whether a bank customer will subscribe to a term deposit (`yes` or `no`) using the Bank Marketing dataset.

The objective is to build a binary classification model, preprocess the numerical and categorical attributes appropriately, and evaluate the model using multiple classification metrics.

## 2. Dataset

The dataset contains **45,211 customer records** and **17 columns**:

* **16 input features**
* **1 target variable (`y`)**

The target variable contains two classes:

* `no`: 39,922 samples (88.30%)
* `yes`: 5,289 samples (11.70%)

### Numerical Features

* age
* balance
* day
* duration
* campaign
* pdays
* previous

### Categorical Features

* job
* marital
* education
* default
* housing
* loan
* contact
* month
* poutcome

The dataset contains no missing (`NaN`) values and no duplicate records.

Some categorical attributes contain the value `unknown`. These values were retained as valid categories rather than removing the corresponding records.

The dataset was provided separately and is **not included in this repository**.

## 3. Methodology

### Data Preparation

The target variable was converted into binary values:

* `no` → `0`
* `yes` → `1`

The dataset was divided into training and testing sets using an **80/20 stratified split**.

* Training samples: 36,168
* Testing samples: 9,043

Stratification was used to maintain approximately the same class distribution in both sets because the target classes are imbalanced.

### Feature Preprocessing

Two preprocessing approaches were applied:

**Numerical features:**

* Standardized using `StandardScaler`

**Categorical features:**

* Converted to numerical form using `OneHotEncoder`
* `handle_unknown="ignore"` was used so that unseen categories would not cause errors during prediction.

A `ColumnTransformer` was used to combine both preprocessing steps.

### Model

The classification model is **Logistic Regression**.

The model was configured with:

* `class_weight="balanced"`
* `max_iter=1000`
* `random_state=42`

`class_weight="balanced"` was used to reduce the effect of the significant class imbalance and improve the model's ability to identify customers who subscribe to a term deposit.

## 4. Model Evaluation

The model was evaluated on the unseen test set.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 84.57% |
| Precision | 41.82% |
| Recall    | 81.47% |
| F1-score  | 55.27% |
| ROC-AUC   | 90.79% |

### Classification Report

| Class | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| No    |    97.19% | 84.98% |   90.68% |
| Yes   |    41.82% | 81.47% |   55.27% |

The model achieved a **ROC-AUC of 0.9079**, indicating strong ability to distinguish between customers who do and do not subscribe.

The recall for the `Yes` class was **81.47%**, meaning the model successfully identified most of the actual term-deposit subscribers.

## 5. Confusion Matrix

The confusion matrix was:

```text
[[6786 1199]
 [ 196  862]]
```

This corresponds to:

* **True Negatives:** 6,786
* **False Positives:** 1,199
* **False Negatives:** 196
* **True Positives:** 862

The model correctly identified **862 of the 1,058 actual subscribers** in the test set.

The relatively low precision for the `Yes` class is partly a result of using balanced class weights, which makes the model more willing to classify customers as potential subscribers in order to reduce missed positive cases.

## 6. Influential Features

The largest Logistic Regression coefficients included:

| Feature            | Coefficient |
| ------------------ | ----------: |
| poutcome = success |     +1.8245 |
| month = mar        |     +1.7220 |
| duration           |     +1.5258 |
| month = jan        |     -1.3026 |
| month = oct        |     +1.2763 |
| contact = unknown  |     -1.0933 |
| month = jul        |     -1.0678 |
| month = sep        |     +0.9848 |
| month = nov        |     -0.9827 |
| month = aug        |     -0.9026 |
| poutcome = unknown |     -0.7869 |
| month = may        |     -0.7252 |
| month = dec        |     +0.7137 |
| poutcome = failure |     -0.6855 |
| job = student      |     +0.6573 |

Positive coefficients increase the model's tendency toward the `yes` class, while negative coefficients increase the tendency toward the `no` class, with other features held constant.

These coefficients represent model associations and should not be interpreted as direct causal relationships.

## 7. Findings

The Logistic Regression model performed well overall, achieving **84.57% accuracy** and **0.9079 ROC-AUC**.

The model was particularly effective at identifying potential term-deposit subscribers, achieving **81.47% recall** for the `Yes` class. However, its precision for this class was lower at **41.82%**, meaning that a considerable number of customers predicted as subscribers did not actually subscribe.

The coefficient analysis showed that previous campaign outcomes, call duration, contact type, month, and certain job categories were among the most influential predictors in the trained model.

## 8. Conclusion

The project demonstrates how Logistic Regression can be applied to a real-world binary classification problem involving both numerical and categorical data.

Using appropriate preprocessing, stratified sampling, one-hot encoding, feature scaling, and balanced class weights produced a model with strong discrimination and good recall for the minority `Yes` class.

The complete implementation is provided in the accompanying Jupyter Notebook.

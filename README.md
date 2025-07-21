# Breast Cancer Classifier

This project presents a machine learning pipeline for classifying breast cancer tumors as **Malignant (M)** or **Benign (B)** based on features extracted from digitized images of fine needle aspirate (FNA) of breast masses.

The workflow includes:
- Exploratory Data Analysis (EDA)
- Feature selection and correlation filtering
- Model training with hyperparameter tuning using cross-validation
- Model evaluation using accuracy and confusion matrices

## 📁 Dataset

The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Data Set**, which can be downloaded as `Breast_canc_data.zip`. It contains:
- 569 samples
- 30  features
- 1 target column: `diagnosis` (M = Malignant, B = Benign)

## 📌 Requirements

You can run the code in **Google Colab**, which has all necessary libraries pre-installed. Otherwise, install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🧠 Models Trained

Five different classifiers were evaluated:

| Model                | Accuracy (%) |
|---------------------|--------------|
| Logistic Regression |   ~97.66     |
| SVC                 |   ~97.66     |
| KNN                 |   ~95.91     |
| Decision Tree       |   ~93.57     |
| Random Forest       |   ~97.08     |

Hyperparameter tuning was performed using **GridSearchCV** and **10-fold Stratified Cross Validation**.

## 📊 Exploratory Data Analysis (EDA)

- Checked for missing values (none found)
- Dropped unnecessary columns (`id`, `Unnamed: 32`)
- Visualized class imbalance (`M` vs `B`) - 357 benign, 212 malignant
- Plotted correlation heatmaps to identify multicollinearity
- Selected top 23 features highly correlated (absolute correlation > 0.3) with the diagnosis (target variable)
- Removed highly correlated features (> 0.9 correlation) to avoid redundancy
- Final model trained on 15 selected features

## 🛠 Pipeline Overview

1. **Data Preprocessing**
   - Loaded and unzipped data
   - Encoded labels (`M`: 1, `B`: 0)
   - Removed highly correlated features

2. **Feature Scaling**
   - Standardized features for SVC, KNN, and Logistic Regression

3. **Model Building**
   - Trained models using `Pipeline` and `GridSearchCV`
   - Selected best hyperparameters based on validation accuracy

4. **Evaluation**
   - Accuracy computed on the test set
   - Confusion matrices plotted for visual comparison

## 📈 Visualization

- **Pair plots** for feature relationships
- **Heatmap** for feature correlation
- **Confusion Matrices** for all models
- **Bar Plot** for accuracy comparison

## 📌 Key Insights

- Logistic Regression and SVC had the highest test accuracy (~97.66%)
- Proper feature selection and avoiding multicollinearity significantly improved model performance
- Scaling was essential for distance-based models like KNN and SVC

## 🔗 Notebook Link

You can access the original notebook here:  
[Colab Notebook](https://colab.research.google.com/drive/1eOKIfYkfwOz08RiNHGHqJcJH2vwseCQr)

## 🔗 Dataset Source

The dataset used in this project is publicly available on Kaggle:
[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

# Titanic Survival Prediction: A Classic End-to-End Classification Project

## 📄 Project Goal

This repository contains a comprehensive, end-to-end solution for the classic "Titanic: Machine Learning from Disaster" competition on Kaggle. The primary objective is to leverage passenger data to build a predictive model that can accurately determine whether a passenger survived the tragedy.

This project serves as a practical demonstration of the entire data science pipeline, from initial data exploration and cleaning to advanced feature engineering, model training, and comparative evaluation.

## ✨ The Workflow: From Raw Data to Actionable Insights

### 1. In-Depth Exploratory Data Analysis (EDA)

The first step was to thoroughly understand the dataset. This involved analyzing each variable for missing values, distributions, and initial correlations with the target variable, `Survived`.

### 2. Advanced Feature Engineering: The Heart of the Project

The key to a successful model in this competition lies in creating powerful features from the raw data. The following features were engineered:
*   **Family Dynamics (`FamilySize` & `IsAlone`):** The `SibSp` (siblings/spouses) and `Parch` (parents/children) columns were combined to create a `FamilySize` feature. This was then used to create a simple but powerful binary feature, `IsAlone`, which proved to have significant predictive power.
*   **Title Extraction from Names:** The passenger's title (e.g., `Mr`, `Mrs`, `Miss`, `Master`) was extracted from the `Name` column using regular expressions. This uncovered hidden social and demographic information, as titles are strongly correlated with age, gender, and survival rate.
*   **Handling Missing Data Intelligently:**
    *   **Age:** Missing `Age` values were imputed, and the continuous variable was then binned into logical `AgeGroup` categories (e.g., 'Baby', 'Child', 'Adult').
    *   **Cabin:** Instead of discarding this feature with many missing values, it was transformed into a binary `InCabin` variable, capturing the signal of whether a passenger had an assigned cabin or not.
*   **Discretization of Continuous Variables:** The `Fare` column was binned into `FareGroup` categories to better capture its relationship with survival without being skewed by extreme outliers.

### 3. Model Bake-Off: A Comparative Analysis

To find the best-performing algorithm, a "bake-off" of five different classification models was conducted:
1.  Logistic Regression
2.  Decision Trees
3.  Random Forest
4.  k-Nearest Neighbors (k-NN)
5.  Support Vector Machines (SVM)

### 4. Robust Model Evaluation

Each model was evaluated using the **Area Under the ROC Curve (AUC)** as the primary performance metric, which is excellent for binary classification problems. Confusion matrices were also used to assess precision and recall.

## 🏆 Final Model & Performance

The **Support Vector Machines (SVM)** model emerged as the top performer, achieving the best balance of predictive power and generalization.

*   **Best Model:** Support Vector Machines
*   **Test Set AUC Score:** **0.82**

This demonstrates a strong ability to distinguish between passengers who survived and those who did not. The analysis of feature importances also confirmed that the engineered features, particularly `Title` and `Sex`, were among the most critical predictors.

![Classifier AUC Comparison](img/auc_comparison.png)
*(Te recomiendo mucho que guardes el gráfico de barras final de tu notebook y lo añadas aquí).*

## 💻 Technologies Used

*   **Language:** Python 3
*   **Core Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
*   **Environment:** Jupyter Notebook

## 🚀 Getting Started

1.  **Clone the repository.**
2.  Install the required Python libraries.
3.  Open the `Who Survived the Titanic?.ipynb` notebook and run the cells to follow the complete data science workflow from start to finish.

## 👤 Author

**Antonio Barrera Mora**

*   **LinkedIn:** https://www.linkedin.com/in/anbamo/
*   **GitHub:** @Kamaranis

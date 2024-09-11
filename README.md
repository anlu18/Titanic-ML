

---

# Titanic Survival Prediction using Machine Learning

## Project Overview

This project was developed by **An Lu**, **Jose Vela**, and **Jim Lao**. Our primary objective was to build a machine learning model to accurately predict the survival of Titanic passengers based on various features such as age, gender, and class.

## Data Source

The dataset for this project was sourced from [Kaggle's Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition.

## Project Documentation

- **Database Type**: Relational (SQL)
- **Data Cleaning and Processing**: Jupyter Notebook
- **Key Packages Used**:
  - Data manipulation and analysis: `pandas`, `numpy`
  - Data visualization: `seaborn`, `matplotlib`
  - Machine Learning Model: `RandomForestClassifier`
- **Data Loading Tools**: pgAdmin, SQLAlchemy

## Project Outline

### 1. Loading the Data

- We began by extracting the Titanic dataset from Kaggle.
- The data was loaded into PostgreSQL tables using pgAdmin.
- We then imported the data into a Pandas DataFrame using SQLAlchemy for further processing and analysis.
- We imported the test csv and train csv.

### 2. Data Cleaning

- We identified missing values within the dataset, particularly in the 'Age' column.
- To handle missing values, we calculated the average age of passengers and used this value to fill in the missing entries.
- We transformed and scaled the "Age" and "Fare" from the titanic_test_df

### 3. Building the Machine Learning Models

- **Data Preparation**:
  - The dataset was scaled and split into training and testing sets to prepare it for modeling.


- **Feature Selection**:
  - The dataset was cleaned by removing irrelevant columns and the target variable.
  - Features used in the model:
    - `Pclass`
    - `Age`
    - `SibSp`
    - `Parch`
    - `Fare`
  
  The following columns were dropped:
  - `survived` (target variable)
  - `sex`, `ticket`, `embarked`, `name` (considered irrelevant or categorical variables that were not encoded)

- **Feature Matrix and Target Variable**:
  - `X`: Features matrix, excluding the target variable and unnecessary columns.
  - `y`: Target variable, which represents the survival status of the passengers.

```python
X = titanic_train_df.drop(columns=["survived", "sex", "ticket", "embarked", "name"]).values
y = titanic_train_df["survived"].values
```

- **Data Scaling**:
  - A `StandardScaler` was applied to scale the features, ensuring that the model converges more quickly and accurately during training.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Model Architecture

- **Deep Learning Model**:
  - A sequential deep learning model was constructed using TensorFlow/Keras.
  - The model architecture includes:
    - **Input Layer**: Takes in the scaled features with an input shape corresponding to the number of selected features.
    - **Hidden Layer**: A fully connected layer with 16 units and a ReLU activation function.
    - **Output Layer**: A single unit with a tanh activation function to predict survival probability.

```python
nn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=16, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(units=1, activation="tanh")
])
```

### Next Steps

- **Model Training**:
  - The model should be compiled with an appropriate optimizer and loss function.
  - Train the model using the scaled feature set and target variable.

- **Evaluation**:
  - After training, the model’s performance should be evaluated on the test set to measure its accuracy and other relevant metrics.

---


  
- **Model Development**:
  - We defined and trained a RandomForest model as our baseline machine learning model.
  - Subsequently, we developed a deep learning model to enhance the prediction accuracy.
  
- **Model Evaluation**:
  - Both models were evaluated on the test dataset to measure their performance and accuracy.

## Conclusion

Through this project, we successfully developed machine learning models capable of predicting the survival chances of Titanic passengers with reasonable accuracy. The combination of data cleaning, feature engineering, and model tuning played a crucial role in achieving these results.

---


  

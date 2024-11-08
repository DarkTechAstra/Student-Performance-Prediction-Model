# Student Performance Prediction Model

This project predicts student exam scores based on various personal, academic, and environmental factors. Using Machine Learning models like Linear Regression, Random Forest, and XGBoost, the project helps in understanding the impact of factors like study hours, parental involvement, extracurricular activities, and teacher quality on student performance.

## Table of Contents
- [Project Overview](#project-overview)
- [How the Model Works](#how-the-model-works)
- [Installation](#installation)
- [Usage](#usage)
  - [Individual Prediction](#individual-prediction)
  - [Batch Prediction](#batch-prediction)
- [Features](#features)
- [Benefits](#benefits)
- [Technologies Used](#technologies-used)

## Project Overview

This project builds a machine learning model to predict the exam scores of students based on a range of input factors. These factors include:

- Hours Studied
- Attendance
- Parental Involvement
- Access to Resources
- Extracurricular Activities
- Sleep Hours
- Previous Scores
- Motivation Level
- Tutoring Sessions
- Teacher Quality
- Physical Activity
- Learning Disabilities

### Key Objectives:
- Help educators and school administrators identify students at risk of poor performance.
- Provide insights into how various factors contribute to a student's academic success.
- Offer personalized recommendations to improve student outcomes.

## How the Model Works

### Data Preprocessing
The dataset contains categorical and numerical features. We clean and preprocess the data by:
- Filling missing values (e.g., filling missing `Teacher_Quality` data with "Medium").
- Converting categorical features like `Low`, `Medium`, and `High` into numerical values (`1`, `2`, `3` respectively).
- Converting binary categorical features (`Yes`/`No`) into numeric values (`1`/`0`).

### Machine Learning Models
The project explores different models to predict exam scores:
- **Linear Regression**: A simple and interpretable model that fits a linear relationship between the input features and exam scores.
- **Random Forest Regressor**: An ensemble method that builds multiple decision trees and averages their results to improve prediction accuracy.
- **XGBoost Regressor**: A powerful and efficient implementation of gradient boosting used for higher accuracy with faster computation.

### Model Performance
The models are evaluated using metrics like **Mean Squared Error (MSE)** and **R-squared (R²)** values to assess how well they predict student performance. Grid Search and cross-validation techniques are used to fine-tune the models.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.comDarkTechAstra/Student-Performance-Prediction.git
   cd StudentPerformancePrediction
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the trained model (`student_performance_model.pkl`) saved in the `models/` directory. If the model isn't present, you can retrain it using the provided notebook or script.

4. Start the Flask application:

   ```bash
   python app.py
   ```

## Usage

The model can be used for both **Individual Prediction** (predicting the exam score of a single student) and **Batch Prediction** (predicting the scores for multiple students from a CSV file).

### Individual Prediction
1. Open the application in your browser by navigating to `http://localhost:5000`.
2. Select the **Individual Prediction** form.
3. Enter the required student data:
   - Hours Studied, Attendance, Parental Involvement, Access to Resources, etc.
4. Click on **Predict**.
5. The predicted exam score will be displayed on the results page.

### Batch Prediction
1. Upload a CSV file with the same feature names as the model expects:
   - Columns: `Hours_Studied`, `Attendance`, `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Sleep_Hours`, `Previous_Scores`, `Motivation_Level`, `Tutoring_Sessions`, `Teacher_Quality`, `Physical_Activity`, `Learning_Disabilities`.
2. Once the file is uploaded, the app will predict exam scores for all students in the file and provide a downloadable CSV file containing the results.

## Features

- **Input Data**: The model uses 12 factors (features) to predict student exam scores. These factors represent a mix of academic, environmental, and behavioral elements.
  
- **Prediction Types**:
  - **Individual Prediction**: Users can enter the data for a single student and get an instant prediction.
  - **Batch Prediction**: Users can upload a CSV file with multiple students' data and get exam scores for all of them.
  
- **Models**: The project includes multiple models for comparison, including Linear Regression, Random Forest, and XGBoost.

## Benefits

- **Early Intervention**: By predicting exam performance in advance, educators can identify students who are struggling and provide them with personalized interventions.
  
- **Data-Driven Insights**: The model provides actionable insights into how various factors like study hours, sleep, and parental involvement affect a student’s academic performance.
  
- **Customizable**: The project is modular and can easily be extended to include more features or other machine learning models.
  
- **Batch Predictions**: The ability to predict exam scores for multiple students in one go makes it practical for large-scale use by schools and institutions.

## Technologies Used

- **Python**: The main language used to build the project.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For building and evaluating machine learning models.
- **XGBoost**: For the gradient boosting algorithm.
- **Flask**: For building the web interface for individual and batch predictions.
- **Pickle**: To save and load the trained machine learning models.
- **Google Colab**: Used for training and experimenting with the models.

from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('models/student_performance_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

def transform_categorical_features(df):
    """
    Transforms categorical features in the DataFrame to numeric values.
    """
    try:
        # Define mappings for categorical features
        mappings = {
            'Parental_Involvement': {'Low': 1, 'Medium': 2, 'High': 3},
            'Access_to_Resources': {'Low': 1, 'Medium': 2, 'High': 3},
            'Extracurricular_Activities': {'No': 0, 'Yes': 1},
            'Motivation_Level': {'Low': 1, 'Medium': 2, 'High': 3},
            'Teacher_Quality': {'Low': 1, 'Medium': 2, 'High': 3},
            'Learning_Disabilities': {'No': 0, 'Yes': 1}
        }

        # Apply transformations
        for column, mapping in mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping)
        
        # Check for unmapped values (NaNs)
        for column in df.columns:
            if df[column].isnull().any():
                return df, f"Error: Unmapped categorical values in column {column}"

        return df, None
    
    except Exception as e:
        return df, str(e)




# Home page route
@app.route('/')
def index():
    return render_template('index.html')

# Individual prediction route
@app.route('/predict-individual', methods=['GET', 'POST'])
def predict_individual():
    if request.method == 'POST':
        try:
            # Extract form data
            form_data = {
                'Hours_Studied': float(request.form['hours_studied']),
                'Attendance': float(request.form['attendance']),
                'Parental_Involvement': request.form['parental_involvement'],
                'Access_to_Resources': request.form['access_to_resources'],
                'Extracurricular_Activities': request.form['extracurricular_activities'],
                'Sleep_Hours': float(request.form['sleep_hours']),
                'Previous_Scores': float(request.form['previous_scores']),
                'Motivation_Level': request.form['motivation_level'],
                'Tutoring_Sessions': float(request.form['tutoring_sessions']),
                'Teacher_Quality': request.form['teacher_quality'],
                'Physical_Activity': float(request.form['physical_activity']),
                'Learning_Disabilities': request.form['learning_disabilities']
            }

            # Convert form data to a DataFrame
            form_df = pd.DataFrame([form_data])

            # Apply categorical transformation
            form_df, error = transform_categorical_features(form_df)
            if error:
                return f"Error: {error}", 400

            # Check for NaNs
            if form_df.isnull().values.any():
                return "Error: Unmapped categorical values. Please check your inputs.", 400

            # Make the prediction
            prediction = model.predict(form_df)
            return render_template('individual_result.html', prediction=prediction[0])
        
        except KeyError as e:
            return f"Error: Missing form field {str(e)}", 400
        except ValueError as e:
            return f"Error: {str(e)}", 400
        except Exception as e:
            return f"Error: {str(e)}", 500
    
    return render_template('individual_input.html')



    
# Batch prediction route
@app.route('/predict-batch', methods=['GET', 'POST'])
def predict_batch():
    if request.method == 'POST':
        file = request.files['csv_file']
        df = pd.read_csv(file)

        # List of expected column names
        expected_columns = [
            'Hours_Studied', 'Attendance', 'Parental_Involvement', 'Access_to_Resources',
            'Extracurricular_Activities', 'Sleep_Hours', 'Previous_Scores', 'Motivation_Level',
            'Tutoring_Sessions', 'Teacher_Quality', 'Physical_Activity', 'Learning_Disabilities'
        ]

        # Check if the columns in the CSV match the expected columns
        if not all(col in df.columns for col in expected_columns):
            missing_columns = [col for col in expected_columns if col not in df.columns]
            return f"Error: Missing columns in the CSV file: {', '.join(missing_columns)}", 400

        # Check for any NaN values in the input data
        if df.isnull().values.any():
            return "Error: The input data contains NaN values. Please check the CSV for missing or invalid data.", 400

        # Make predictions for the batch data
        predictions = model.predict(df)
        df['Predicted_Score'] = predictions
        df.to_csv('predictions.csv', index=False)

        return send_file('predictions.csv', as_attachment=True, download_name='predictions.csv')

    return render_template('batch_input.html')
### just a message for me friend emman to clarify how it works
if __name__ == '__main__':
    app.run(debug=True)

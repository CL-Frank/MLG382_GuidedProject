import os


import xgboost as xgb
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from dash import Dash, html, dcc, Input, Output
from functools import lru_cache
import dash_bootstrap_components as dbc

@lru_cache(maxsize=1)
def get_DLmodel():
    print("Loading deep learning model...")
    return load_model(r'../artifacts/dl_model.h5')

with open('../notebooks/features.pkl', 'rb') as f:
    features = pickle.load(f)

with open('../notebooks/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open(r'../artifacts/rf_model.pkl', 'rb') as f:
    randomforest_model = joblib.load(f)

with open(r'../artifacts/lr_model.pkl', 'rb') as f:
    regression_model = joblib.load(f)

with open(r'../artifacts/xgb_model.pkl', 'rb') as f:
    xgboost_model = joblib.load(f)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Sample input layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Student Grade Predictor", className="text-center mb-4"), width=12)
    ]),
    
    dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardBody([

                    dbc.Label("Age", className="text-center w-100"),
                    dbc.Input(id='age', type='number', value=18, className="mb-3"),

                    dbc.Label("Gender", className="text-center w-100"),
                    dcc.Dropdown(
                        id='gender',
                        options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}],
                        value=0,
                        className="mb-3"
                    ),

                    dbc.Label("Study Time Weekly", className="text-center w-100"),
                    dbc.Input(id='study_time', type='number', value=15, className="mb-3"),

                    dbc.Label("Absences", className="text-center w-100"),
                    dbc.Input(id='absences', type='number', value=5, className="mb-3"),

                    dbc.Label("GPA", className="text-center w-100"),
                    dbc.Input(id='gpa', type='number', value=2, className="mb-3"),

                    dbc.Label("Ethnicity", className="text-center w-100"),
                    dcc.Dropdown(
                        id='ethnicity',
                        options=[
                            {'label': 'Caucasian', 'value': 0},
                            {'label': 'African American', 'value': 1},
                            {'label': 'Asian', 'value': 2},
                            {'label': 'Other', 'value': 3},
                        ],
                        value=0,
                        className="mb-3"
                    ),

                    dbc.Label("Parental Education", className="text-center w-100"),
                    dcc.Dropdown(
                        id='parental_education',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'High School', 'value': 1},
                            {'label': 'Some College', 'value': 2},
                            {'label': 'Bachelors', 'value': 3},
                            {'label': 'Higher Study', 'value': 4}
                        ],
                        value=0,
                        className="mb-3"
                    ),

                    dbc.Label("Parental Support", className="text-center w-100"),
                    dcc.Dropdown(
                        id='parental_support',
                        options=[
                            {'label': 'None', 'value': 0},
                            {'label': 'Low', 'value': 1},
                            {'label': 'Moderate', 'value': 2},
                            {'label': 'High', 'value': 3},
                            {'label': 'Very High', 'value': 4}
                        ],
                        value=0,
                        className="mb-3"
                    ),
                    

                    html.Div([
                        dbc.Label("Activities"),
                        dbc.Checklist(
                            options=[{'label': 'Tutoring', 'value': 1}],
                            value=[], id='tutoring', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Extracurricular', 'value': 1}],
                            value=[], id='extracurricular', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Sports', 'value': 1}],
                            value=[], id='sports', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Music', 'value': 1}],
                            value=[], id='music', inline=True
                        ),
                        dbc.Checklist(
                            options=[{'label': 'Volunteering', 'value': 1}],
                            value=[], id='volunteering', inline=True
                        ),
                    ], className="mb-4"),

                    dbc.Button("Predict", id='predict_button', color="primary", className="mb-3 w-100"),

                    html.Div(id='prediction-output', className="mt-3 text-center")

                ])
            ], className="p-4 shadow-sm border rounded bg-white"),  # Card styling
            width=4,
            className="mx-auto"
        )
    ])
], fluid=True)

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict_button', 'n_clicks'),
     Input('age', 'value'),
     Input('gender', 'value'),
     Input('study_time', 'value'),
     Input('absences', 'value'),
     Input('tutoring', 'value'),
     Input('extracurricular', 'value'),
     Input('sports', 'value'),
     Input('music', 'value'),
     Input('volunteering', 'value'),
     Input('gpa', 'value'),
     Input('ethnicity', 'value'),
     Input('parental_education', 'value'),
     Input('parental_support', 'value')]
)
def predict_grade(n_clicks, age, gender,study_time, absences, tutoring,extracurricular, sports, music, volunteering, ethnicity, parental_education, parental_support, gpa):
    if (n_clicks or 0) > 0 and None not in (age, gender, study_time, absences, ethnicity, parental_education, parental_support, gpa):
        input_data = {
            'Age': [age],
            'Gender': [gender],
            'Ethnicity': [ethnicity],
            'ParentalEducation': [parental_education],
            'StudyTimeWeekly': [study_time],
            'Absences': [absences],
            'Tutoring': [1 if tutoring and 1 in tutoring else 0],
            'ParentalSupport': [parental_support],
            'Extracurricular': [1 if extracurricular and 1 in extracurricular else 0],
            'Sports': [1 if sports and  1 in sports else 0],
            'Music': [1 if music and 1 in music else 0],
            'Volunteering': [1 if volunteering and 1 in volunteering else 0],
            'GPA': [gpa],
        }
         
        input_df = pd.DataFrame(input_data)
        #  print(f'Input Features{input_df.columns}')
        #  print(f'Expected Features{features}')

        input_df['StudyTimePerAbsence'] = input_df['StudyTimeWeekly'] / (input_df['Absences'] + 1)
        input_df['TotalExtracurricular'] = input_df[['Extracurricular', 'Sports', 'Music', 'Volunteering']].sum(axis=1)

        #  print(f'NEW Features{input_df.columns}')
         
        bins = [0, 5, 10, 15, 20]
        labels = ['Low', 'Moderate', 'High', 'Very High']
        input_df['StudyTimeCategory'] = pd.cut(input_df['StudyTimeWeekly'], bins=bins, labels=labels, include_lowest=True)

        categorical_cols = ['Ethnicity', 'ParentalEducation', 'StudyTimeCategory']
        data_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

         # Add missing dummy columns
        for col in features:
           if col not in data_encoded.columns:
                data_encoded[col] = 0  # or False for boolean features

        data_encoded = pd.DataFrame(data_encoded[features])
        #  print(f'NEW 2 Features{data_encoded.columns}')

         
        num_features = ['Age','StudyTimeWeekly', 'Absences', 'GPA', 'StudyTimePerAbsence', 'TotalExtracurricular']
        data_encoded[num_features] = scaler.transform(data_encoded[num_features])

        #  for col in features:
        #     if col not in input_df.columns:
        #      input_df[col] = 0
        #  input_df = input_df[features]  # reorder columns to match

 
        output_df = data_encoded


        #  print(f'Output Features{data_encoded.columns}')
        print(output_df)



    
        # Deep learning model prediction
        print("Loading model...")
        DL_model = get_DLmodel()
        print("Model loaded!")

        dl_prediction = DL_model.predict(output_df)
        rf_prediction = randomforest_model.predict_proba(output_df)  # Using predict_proba to get probabilities
        logreg_prediction = regression_model.predict_proba(output_df)  # Using predict_proba to get probabilities
        xgboost_prediction = xgboost_model.predict_proba(output_df)  # Using predict_proba to get probabilities
        print("Prediction done")

        # Deep Learning model - Get grade and confidence
        if len(dl_prediction.shape) == 2 and dl_prediction.shape[1] > 1:
            dl_class_prediction = np.argmax(dl_prediction)
            dl_confidence = np.max(dl_prediction)
        else:
            dl_class_prediction = int(round(float(dl_prediction[0][0])))
            dl_confidence = float(dl_prediction[0][0]) if dl_class_prediction == 1 else 1 - float(dl_prediction[0][0])

        # Map the numeric prediction to letter grades
        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: "F"}  # Update as per your model's class labels
        dl_class_prediction_grade = grade_map.get(dl_class_prediction, 'Unknown')

        # Random Forest prediction
        rf_class_prediction = np.argmax(rf_prediction)  # Get the index of the predicted class
        rf_confidence = np.max(rf_prediction)  # Get the confidence (probability) of the predicted class

        # Logistic Regression prediction
        logreg_class_prediction = np.argmax(logreg_prediction)  # Get the index of the predicted class
        logreg_confidence = np.max(logreg_prediction)  # Get the confidence (probability) of the predicted class

        # XGBoost prediction
        xgboost_class_prediction = np.argmax(xgboost_prediction)  # Get the index of the predicted class
        xgboost_confidence = np.max(xgboost_prediction)  # Get the confidence (probability) of the predicted class

        # Prepare for the table display
        predictions = [
            ("Deep Learning", dl_class_prediction_grade, dl_confidence * 100),
            ("Random Forest", grade_map.get(rf_class_prediction, 'Unknown'), rf_confidence * 100),
            ("Logistic Regression", grade_map.get(logreg_class_prediction, 'Unknown'), logreg_confidence * 100),
            ("XGBoost", grade_map.get(xgboost_class_prediction, 'Unknown'), xgboost_confidence * 100)
        ]
         
        

        return dbc.Card([
            dbc.CardHeader("Model Predictions", className="bg-success text-white text-center"),
            dbc.CardBody([
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Model", className="text-center"),
                        html.Th("Prediction", className="text-center"),
                        html.Th("Confidence", className="text-center")
                    ])),
                    html.Tbody([html.Tr([
                        html.Td(model, className="text-center"),
                        html.Td(grade, className="text-center"),
                        html.Td(f"{confidence:.2f}%", className="text-center")
                    ]) for model, grade, confidence in predictions])
                ], bordered=True, striped=True, hover=True, responsive=True)
            ])
        ], className="mt-4 shadow-sm")

    return "Please fill in all fields."
        
if __name__ == "__main__":
    print("Launching Dash app...")
    try:
        # app.run(debug=True, host='127.0.0.1', port=8050)
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print("Failed to start server:", e)
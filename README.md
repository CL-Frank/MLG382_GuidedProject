# MLG382_GuidedProject
## Student Grade Prediction Platform
An interactive machine learning and deep learning application for predicting student grade classifications (A–F) based on demographic and academic-related features. Built with Dash and deployed on Render, the platform enables users to explore multiple predictive models and visualize results in real-time.


This project uses multiple classification models—including Logistic Regression, Random Forest, XGBoost, and a Deep Learning Neural Network—to predict a student's grade class from features such as age, gender, study time, absences, and parental support.

### Key Features
- Predicts student grade class (A–F) using trained models.

- Displays predictions from multiple models side-by-side with confidence scores.

- Built with Dash and styled with Dash Bootstrap Components for a clean UI.

- Hosted on Render for public access.

- Includes feature engineering and preprocessing consistent with model training.


### Models Used
- Logistic Regression
- Random Forest
- XGBoost
- Deep Learning (Keras Sequential Model)

Models are trained and evaluated using scikit-learn and TensorFlow/Keras, and are saved in the artifacts/ directory.

Model	Accuracy	Precision	Recall	F1 Score  
Logistic Regression	0.6722	0.6353	0.6722	0.6495  
Random Forest	0.6806	0.6660	0.6806	0.6610  
XGBoost	0.6806	0.6618	0.6806	0.6644  
Deep Learning	0.7300	0.7100	0.7300	0.7100  

### Future Work
- Add Explainable AI (XAI) using SHAP or LIME for transparency.
- Improve class balance through data augmentation or SMOTE.
- Expand feature set with historical grades, behavioral metrics, or attendance logs.
- Enable CSV batch upload for bulk predictions.

### Authors
Team Name: Group H
Contributors:

Bianca Grobler 600537  
Adolph Jacobus van Coller 601005  
Caydan Frank 578131  
Renaldo Jardim 601333  

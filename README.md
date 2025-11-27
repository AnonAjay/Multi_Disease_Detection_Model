<<<<<<< HEAD
🩺 LiveGuard Disease Classifier WebApp
Modern web application for rapid disease classification—including Breast Cancer, Heart Disease, Kidney Disease, and Diabetes. Designed for easy use, fast inference, and robust deployment.

Features
Fast tabular disease prediction for multiple conditions

Breast Cancer: Upload CSV/JSON or fill fields

Heart Disease: Simple feature form

Kidney Disease: Optimized, multi-field feature form

Diabetes: Numeric form for instant risk assessment

Optimized ML pipelines—feature engineering, scaling, and encoding included for robust predictions

Responsive web UI—dark mode, loading overlays, animated results

AJAX-powered forms for fast, interactive prediction

Technologies
Backend: Python 3.x + Flask + scikit-learn

Frontend: HTML/CSS/JS (vanilla, no framework)

Models: Logistic Regression, Ridge Classifier; full preprocessing inside pipelines

Deployment: Local Flask server (app.py), easy to upgrade for cloud/VPS hosting

Setup
Clone this repository:

text
git clone https://github.com/yourusername/liveguard-disease-classifier.git
cd liveguard-disease-classifier
Install dependencies:

text
conda env create -f environment.yml
conda activate allpurpose
or

text
pip install -r requirements.txt
Train or download models:

You can retrain using the provided Python scripts in /model_scripts, or use pre-trained .pkl files in /trained_models.

Example:

text
# To retrain:
python model_scripts/train_breast_cancer.py
python model_scripts/train_heart_disease.py
python model_scripts/train_kidney_disease.py
python model_scripts/train_diabetes.py
Run the server:

text
python app.py
This will start the Flask server at http://localhost:5000

Usage
Web Page:
Go to http://localhost:5000 and select the tabular predictor you wish to use.

Uploading:

Tabular predictors support CSV/JSON uploads and direct form entry.

Example: Upload heart_disease_sample.csv or enter features in the form.

Results:
Fast prediction with animated loading and slide-down diagnosis, including explanations and disease information.

Backend API:
Endpoints accept POST with JSON body { "features": { ... }}; see /app.py for formats and documentation.

Directory Structure
text
/static/
    /styles/      # CSS files
    /images/      # Logos and UI images
/templates/
    breast_cancer.html
    heart_disease.html
    kidney_disease.html
    diabetes.html
/model_scripts/
    train_breast_cancer.py
    train_heart_disease.py
    train_kidney_disease.py
    train_diabetes.py
/trained_models/
    breast_cancer_pipeline.pkl
    heart_logreg_pipeline.pkl
    kidney_logreg_pipeline.pkl
    ridge_classifier_diabetes_pipeline.pkl
app.py
README.md
requirements.txt
environment.yml
Sample Data
Check /data/ for sample CSV input files and templates.

Column order and input format must match those provided in forms and scripts.

Customization & Extending
Add more diseases or predictors by duplicating a template and training script.

Pipelines handle feature engineering and encoding—no extra backend code needed!

Deploy to production with Gunicorn and Nginx, or use Docker for containerization.

License
MIT License © [Your Name or Company]

For improvements or issues, open an issue or PR. Contributions welcome!
=======
# Multi_Disease_Detection_Model
General_Purpose_Disease-Classification is a Flask-based multi-disease detection app that combines nine ML/CNN models on Kaggle datasets to analyze medical images and tabular data, providing unified predictions for multiple conditions via image upload and form/CSV/JSON inputs.
>>>>>>> 41ba26a9bc9bd9c5dd289e43f1987c49b7f48572

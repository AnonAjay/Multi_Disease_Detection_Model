import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io

app = Flask(__name__)

# Load tabular models (update paths as needed)
pipeline_paths = {
    "breast_cancer_pipeline": "trained_models/breast_cancer_pipeline.pkl",
    "heart_logreg_pipeline": "trained_models/heart_logreg_pipeline.pkl",
    "kidney_logreg_pipeline": "trained_models/kidney_logreg_pipeline.pkl",
    "ridge_classifier_diabetes_pipeline": "trained_models/ridge_classifier_diabetes_pipeline.pkl"
}
tab_models = {name: joblib.load(path) for name, path in pipeline_paths.items()}

# Load CNN .h5 models (update paths as needed)
cnn_paths = {
    "braintumor": "trained_models/cnn_brain_tumor_best.h5",
    "breasttumor": "trained_models/cnn_breast_tumor_best.h5",
    "lungcancer": "trained_models/lung_cancer_best.h5",
    "pneumonia": "trained_models/pneumonia_cnn_model.h5",
    "gastro": "trained_models/gastro_cnn_best.h5"
}
cnn_models = {name: load_model(path) for name, path in cnn_paths.items()}

def preprocess_img(imgfile, target_size=(224,224)):
    img = image.load_img(io.BytesIO(imgfile.read()), target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    return np.expand_dims(x, axis=0)


@app.route('/')
def index():
    return render_template("index.html")

############ CNN IMAGE ENDPOINTS ############

@app.route('/predict/brain_tumor', methods=['GET', 'POST'])
def predict_braintumor():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        imgfile = request.files['file']
        x = preprocess_img(imgfile)
        model = cnn_models['braintumor']
        pred_probs = model.predict(x)[0]
        class_idx = np.argmax(pred_probs)
        classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
        prediction = classes[class_idx]
        return jsonify({"prediction": prediction})
    else:
        return render_template("brain_tumor.html")

@app.route('/predict/breast_tumor', methods=['GET', 'POST'])
def predict_breasttumor():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        imgfile = request.files['file']
        x = preprocess_img(imgfile)
        model = cnn_models['breasttumor']
        pred_probs = model.predict(x)[0]
        class_idx = np.argmax(pred_probs)
        classes = ["Malignant", "Benign", "No Tumor", "Other"]
        prediction = classes[class_idx]
        return jsonify({"prediction": prediction})
    else:
        return render_template("breast_tumor.html")

@app.route('/predict/lung_cancer', methods=['GET', 'POST'])
def predict_lungcancer():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        imgfile = request.files['file']
        x = preprocess_img(imgfile)
        model = cnn_models['lungcancer']
        pred_probs = model.predict(x)[0]
        class_idx = np.argmax(pred_probs)
        classes = ["Lung Cancer", "Normal", "Other"]
        prediction = classes[class_idx]
        return jsonify({"prediction": prediction})
    else:
        return render_template("lung_cancer.html")

@app.route('/predict/pneumonia', methods=['GET', 'POST'])
def predict_pneumonia():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        imgfile = request.files['file']
        x = preprocess_img(imgfile)
        model = cnn_models['pneumonia']
        pred_probs = model.predict(x)[0]
        class_idx = np.argmax(pred_probs)
        classes = ["Pneumonia Detected", "Normal"]
        prediction = classes[class_idx]
        return jsonify({"prediction": prediction})
    else:
        return render_template("pneumonia.html")

@app.route('/predict/gastro', methods=['GET', 'POST'])
def predict_gastro():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        imgfile = request.files['file']
        x = preprocess_img(imgfile)
        model = cnn_models['gastro']
        pred_probs = model.predict(x)[0]
        class_idx = np.argmax(pred_probs)
        classes = ["Ulcer", "Polyps", "Normal", "Other"]
        prediction = classes[class_idx]
        return jsonify({"prediction": prediction})
    else:
        return render_template("gastro_disease.html")

############ TABULAR ENDPOINTS ############

@app.route('/predict/breast_cancer', methods=['GET', 'POST'])
def predict_breast_cancer_tabular():
    if request.method == 'POST':
        data = request.get_json()
        arr = pd.Series(data['features'])
        area_mean_log       = np.log1p(arr[3])
        concavity_mean_sqrt = np.sqrt(arr[6])
        texture_ratio       = arr[1] / (arr[21] + 1)
        features_full = arr.tolist() + [area_mean_log, concavity_mean_sqrt, texture_ratio]
        df_pred = pd.DataFrame([features_full])
        model = tab_models['breast_cancer_pipeline']
        pred = model.predict(df_pred)
        diagnosis = "Malignant" if int(pred[0]) == 1 else "Benign"
        return jsonify({"prediction": diagnosis})
    else:
        return render_template("breast_cancer.html")

@app.route('/predict/heart', methods=['GET', 'POST'])
def predict_heart():
    if request.method == 'POST':
        data = request.get_json()
        features_dict = data['features']
        df_pred = pd.DataFrame([features_dict])
        model = tab_models['heart_logreg_pipeline']
        pred = model.predict(df_pred)
        result = "Heart Disease Detected" if int(pred[0]) == 1 else "No Heart Disease"
        return jsonify({"prediction": result})
    else:
        return render_template("heart_disease.html")

@app.route('/predict/kidney', methods=['GET', 'POST'])
def predict_kidney():
    if request.method == 'POST':
        data = request.get_json()
        features_dict = data['features']
        df_pred = pd.DataFrame([features_dict])
        model = tab_models['kidney_logreg_pipeline']
        pred = model.predict(df_pred)
        result = "Chronic Kidney Disease" if int(pred[0]) == 1 else "Not Chronic Kidney Disease"
        return jsonify({"prediction": result})
    else:
        return render_template("kidney_disease.html")

@app.route('/predict/diabetes', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'POST':
        data = request.get_json()
        features = data['features']
        features['BMI_Age_Ratio'] = features['BMI'] / (features['Age'] + 1)
        features['Glucose_Pedigree'] = features['Glucose'] * features['DiabetesPedigreeFunction']
        features['BP_Insulin'] = features['BloodPressure'] * features['Insulin']
        df_pred = pd.DataFrame([features])
        model = tab_models['ridge_classifier_diabetes_pipeline']
        pred = model.predict(df_pred)
        result = "Diabetes Detected" if int(pred[0]) == 1 else "No Diabetes"
        return jsonify({"prediction": result})
    else:
        return render_template("diabetes.html")

@app.errorhandler(404)
def page_not_found(e):
    return "404 Not Found", 404

if __name__ == '__main__':
    app.run(debug=True)

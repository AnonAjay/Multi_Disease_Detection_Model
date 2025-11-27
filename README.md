# 🩺 Multi Disease Detection Model (LiveGuard)

Multi_Disease_Detection_Model (LiveGuard) is a Flask-based multi-disease detection web app that combines multiple ML and CNN models trained on Kaggle datasets to analyze both medical images and tabular clinical data. The app provides unified predictions for conditions such as breast cancer, heart disease, kidney disease, diabetes, and more via image upload and form/CSV/JSON inputs. [web:36][web:225]

---

## Features

- Fast tabular disease prediction for multiple conditions (breast cancer, heart disease, kidney disease, diabetes, etc.).
- Image-based prediction for supported diseases using CNN models.
- Multiple input modes:
  - CSV / JSON upload.
  - Direct HTML forms.
  - Image upload (X-ray / scan, depending on model).
- Optimized ML pipelines with preprocessing (scaling, encoding, feature engineering) baked into each model.
- Responsive web UI with dark mode, loading overlays, and animated result sections.
- AJAX-powered forms for smooth, interactive prediction without full-page reloads. [web:220][web:221]

---

## Tech Stack

- **Backend**: Python 3.x, Flask, scikit-learn, Keras/TensorFlow.
- **Frontend**: HTML, CSS, vanilla JavaScript.
- **Models**:
  - Tabular: Logistic Regression, Ridge Classifier, and other sklearn models wrapped in pipelines.
  - Imaging: CNN models saved as `.h5` (for supported diseases).
- **Deployment**: Local Flask server (`app.py`), easily extendable to cloud / VPS / Docker. [web:220][web:222]

---

## Setup

### 1. Clone the repository

### 2. Create environment & install dependencies

Using Conda:
    conda env create -f environment.yml (will provide later working on some other project)
    conda activate allpurpose

Or using pip:
    pip install -r requirements.txt (will provide later as well)


### 3. Models

- Use pre-trained `.pkl` / `.h5` models in `/trained_models`, or
- Retrain using scripts in `/model_scripts` (if provided), for example:


> Large datasets are **not** included in this repo; please download them from their original Kaggle sources and update paths in the training scripts accordingly. [web:93][web:102]

### 4. Run the server


The app will start at: `http://localhost:5000`

---

## Usage

- Open `http://localhost:5000` in your browser.
- Choose the required predictor (e.g., Breast Cancer, Heart Disease, Kidney Disease, Diabetes, etc.).
- Provide inputs:
  - Upload CSV/JSON for batch prediction, or
  - Fill the form for single-patient prediction, or
  - Upload an image (for supported imaging models).
- Submit and view the predicted risk/label along with a clean UI response. [web:220][web:225]

---


---

## Extending the App

- Add a new disease:
  - Create a training script in `/model_scripts`.
  - Save the trained model in `/trained_models`.
  - Add a new HTML template and route in `app.py`.
- Because preprocessing is encapsulated in pipelines, adding new tabular models usually requires minimal backend changes. [web:220][web:221]

---

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.




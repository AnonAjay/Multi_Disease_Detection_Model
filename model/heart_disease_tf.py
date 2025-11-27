import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
import joblib

# 1. Load data
df = pd.read_csv("../data/heart_disease_data/heart.csv")

# 2. Basic EDA
plt.figure(figsize=(6,4))
sns.countplot(x="HeartDisease", data=df)
plt.title("Class Distribution")
plt.show()

print("Columns:", df.columns.tolist())
print(df.describe(), "\n")

# 3. One-hot encode categorical features
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['HeartDisease']]

# 4. Spliting the data in X and Y
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)
pipe = Pipeline([
    ('pre', preprocessor),
    ('logreg', LogisticRegression(solver='lbfgs', max_iter=500, random_state=42))
])

# 7. Train and save
pipe.fit(X_train, y_train)
joblib.dump(pipe, "trained_models/heart_logreg_pipeline.pkl")
print("Pipeline saved as trained_models/heart_logreg_pipeline.pkl")

# 8. Evaluate
y_pred = pipe.predict(X_test)
y_pred_probs = pipe.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
auc_score = auc(fpr, tpr)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color="darkorange")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
# Replace this with your actual dataset
data = pd.read_csv("data.csv")

# Feature columns
features = ["age", "bmi", "systolic", "diastolic", "cholesterol", "glucose", "smoking", "alcohol", "activity"]

# Target columns
target_heart = "heart_disease"
target_diabetes = "diabetes"

# Prepare input and output
X = data[features]
y_heart = data[target_heart]
y_diabetes = data[target_diabetes]

# Split data
X_train, X_test, y_heart_train, y_heart_test, y_diabetes_train, y_diabetes_test = train_test_split(
    X, y_heart, y_diabetes, test_size=0.2, random_state=42
)

# Train models
heart_model = RandomForestClassifier()
diabetes_model = RandomForestClassifier()

heart_model.fit(X_train, y_heart_train)
diabetes_model.fit(X_train, y_diabetes_train)

# Save both models in one file
joblib.dump([heart_model, diabetes_model], "model.pkl")

print("✅ Models trained and saved as model.pkl")

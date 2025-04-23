import pandas as pd
import numpy as np

# Set number of samples
n_samples = 500

# Generate synthetic health data
np.random.seed(42)

age = np.random.randint(18, 80, size=n_samples)
bmi = np.round(np.random.normal(loc=25, scale=5, size=n_samples), 1)
systolic = np.random.randint(100, 160, size=n_samples)
diastolic = np.random.randint(60, 100, size=n_samples)
cholesterol = np.random.randint(150, 300, size=n_samples)
glucose = np.random.randint(70, 200, size=n_samples)
smoking = np.random.randint(0, 2, size=n_samples)
alcohol = np.random.randint(0, 2, size=n_samples)
activity = np.random.randint(0, 2, size=n_samples)

# Simple logic for synthetic target generation
heart_disease = ((age > 50) & (systolic > 140) & (cholesterol > 220) | (smoking == 1)).astype(int)
diabetes = ((bmi > 28) & (glucose > 140) | (age > 55)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "age": age,
    "bmi": bmi,
    "systolic": systolic,
    "diastolic": diastolic,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "smoking": smoking,
    "alcohol": alcohol,
    "activity": activity,
    "heart_disease": heart_disease,
    "diabetes": diabetes
})

# Save to CSV
df.to_csv("data.csv", index=False)
print("✅ Synthetic dataset generated as 'data.csv' with", n_samples, "records.")

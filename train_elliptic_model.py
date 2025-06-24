import pandas as pd
from sklearn.covariance import EllipticEnvelope
import joblib

# Simulated clean training data (normally distributed)
data = {
    "temperature": [22, 23, 21, 20, 24, 22, 23, 25, 22, 21],
    "humidity": [40, 42, 38, 41, 39, 40, 43, 44, 39, 40]
}

df = pd.DataFrame(data)

# Create the model
model = EllipticEnvelope(contamination=0.1)  # 10% anomalies expected
model.fit(df)

# Save the model
joblib.dump(model, "model.joblib")

print("✅ Model trained and saved as model.joblib")

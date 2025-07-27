import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load training data
df = pd.read_csv("data/sample_grid_data.csv")

# Define features and target
features = ["Voltage", "Current", "Temperature", "Load"]
X = df[features]
y = df["Fault"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open("models/grid_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")

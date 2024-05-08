import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset with specified encoding and error handling
data = pd.read_csv('dataset.csv', encoding='latin1')

# Select relevant features and target variable
features = ['TEMPERATURA', 'VENTOINTENSIDADE', 'FFMC', 'DMC', 'ISI']
target = 'INCENDIO'  # Assuming this column indicates whether a wildfire occurred

# Split the dataset into features and target
X = data[features]
y = data[target]

print(data)

# Normalize features if necessary
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Example new user-provided values
new_values = [[20, 10, 30, 40, 50]]  # Replace with actual values

# Scale the new values
new_values_scaled = scaler.transform(new_values)

# Predict the probability of a wildfire
probability = model.predict_proba(new_values_scaled)[:, 1]
print(f"Probability of a wildfire: {probability[0]}")
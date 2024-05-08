import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import csv  # Import the csv module

# Step 1: Load the Dataset
data = pd.read_csv('simpler.csv', encoding='latin1')

# Reset the index to ensure it's a simple range of integers
data.reset_index(drop=True, inplace=True)

# Step 2: Preprocessing
data = data.dropna()

features = ["TEMPERATURA","HUMIDADERELATIVA","VENTOINTENSIDADE","PRECEPITACAO","FFMC","DMC","ISI"]
target = 'INCENDIO'  # Assuming this column indicates whether a wildfire occurred

# Split the dataset into features and target
X = data[features]
y = data[target]

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train Classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Step 5: Evaluate Classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Initialize CSV writer
with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Index", "Prediction", "Actual"])  # Writing the header

    i = 0
    correct = 0
    incorrect = 0

    done = False

    while True:

        data_size = data.shape[0]-1
        random_number = random.randint(1, data_size)

        selected_entry_features = data.iloc[random_number][features].values.reshape(1, -1)
        incendio_value = data.iloc[random_number]["INCENDIO"]

        # Preserve feature names during prediction
        prediction = classifier.predict(pd.DataFrame(selected_entry_features, columns=features))

        print("===============================================")
        print("(#" + str(i) + ") Prediction from random entry " + str(random_number) + ":")
        print("Prediction:", prediction[0])
        print("Actual:", int(incendio_value))

        i = i + 1

        if prediction[0]!= incendio_value:
            incorrect = incorrect + 1
        else:
            correct = correct + 1

        # Write prediction to CSV
        writer.writerow([i, prediction[0], int(incendio_value)])

        if i == 100000: 
            done = True
            break

if done:
    print("------------------------------------------------")
    print("Correct: " + str(correct))
    print("Incorrect: " + str(incorrect))

    if correct!= 0:
        ratio = correct / i
        print("Computed accuracy: " + str(round(accuracy, 3)))
        print("Actual accuracy: " + str(round(ratio, 3)))
        print("------------------------------------------------")
    else:
        print("------------------------------------------------")

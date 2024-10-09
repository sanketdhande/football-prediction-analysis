import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# This is where you'd load the existing data
df = pd.read_csv('Clean_Model_Ready.csv')

# Preprocessing (similar to before)
df['venue_encoded'] = LabelEncoder().fit_transform(df['venue'])
df['result_encoded'] = LabelEncoder().fit_transform(df['result'])

# Select features and target variable
features = ['xg', 'xga', 'poss', 'sh', 'sot', 'venue_encoded']
X = df[features]
y = df['result_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# Function to take user input
def user_input_prediction():
    print("Enter details for the upcoming match prediction:")
    venue = input("Venue (Home/Away): ")
    xg = float(input("Expected Goals (xG): "))
    xga = float(input("Expected Goals Against (xGA): "))
    poss = float(input("Possession (%): "))
    shots = int(input("Number of Shots: "))
    sot = int(input("Number of Shots on Target: "))

    # Encode 'Venue' input as 0 for Away, 1 for Home
    if venue.lower() == 'home':
        venue_encoded = 1
    else:
        venue_encoded = 0

    # Prepare the feature vector for prediction
    match_data = np.array([[xg, xga, poss, shots, sot, venue_encoded]])

    # Predict the result using the trained model
    prediction = rf_model.predict(match_data)

    # Decode prediction back to W, L, or D
    result = LabelEncoder().fit(df['result']).inverse_transform(prediction)

    print(f"The predicted outcome of the match is: {result[0]}")


# Call the function to take user input and predict
user_input_prediction()
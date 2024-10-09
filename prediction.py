import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('Clean_Model_Ready.csv')

# Preprocessing
# Convert date to datetime format
df['date'] = pd.to_datetime(df['date'])

# Encode categorical variables
label_encoder = LabelEncoder()
df['venue_encoded'] = label_encoder.fit_transform(df['venue'])
df['result_encoded'] = label_encoder.fit_transform(df['result'])  # Target variable

# Select features
features = ['xg', 'xga', 'poss', 'sh', 'sot', 'venue_encoded']
X = df[features]
y = df['result_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
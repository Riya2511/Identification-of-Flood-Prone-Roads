import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load the dataset containing historical flood data and road features
data = pd.read_csv('flood_data.csv')
# Split data into features (X) and labels (y)
X = data[['Elevation', 'Proximity_to_River', 'Rainfall', 'Road_Type']]
y = data['Flood_Prone']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# Make predictions
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
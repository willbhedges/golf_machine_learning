import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data generation
data = {
    'Player': ['Player1', 'Player2', 'Player3', 'Player4', 'Player5'],
    'Score': [72, 76, 78, 70, 75],  # Actual scores
    'Weather': ['Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy'],
    'Course_Difficulty': [3, 4, 5, 3, 4],  # Scale of 1 to 5
    'Driving_Accuracy': [80, 75, 70, 85, 78],  # Percentage
    'Putting_Accuracy': [75, 70, 68, 72, 80],  # Percentage
}

df = pd.DataFrame(data)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['Weather'])

# Split data into features and target
X = df.drop(['Player', 'Score'], axis=1)
y = df['Score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Sample prediction
sample_prediction = model.predict(pd.DataFrame({
    'Course_Difficulty': [3],
    'Driving_Accuracy': [78],
    'Putting_Accuracy': [72],
    'Weather_Cloudy': [1],
    'Weather_Rainy': [0],
    'Weather_Sunny': [0]
}))
print(f'Sample Prediction: {sample_prediction}')

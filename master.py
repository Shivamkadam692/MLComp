import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# Step 1: Load the data
# Use relative path to the Dataset directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Dataset', 'mtcars.csv')  # Using mtcars.csv as it exists in the Dataset folder
data = pd.read_csv(data_path)
# Step 2: Preprocess the data
# Display the first few rows of the dataset
print(data.head())

# Summary statistics
print(data.describe())
# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Impute missing values in a specific column (e.g., 'distance') with the mean
mean_value = data['distance'].mean()
data['distance'].fillna(mean_value, inplace=True)
data.dropna(inplace=True)
# Check for missing values after handling
missing_values = data.isnull().sum()
print(missing_values)
# Preprocess the data
X = data[['speed']]  	# Input feature (in this case, just the speed)
Y = data['distance']  	# Target variable (stopping distance)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Step 4: Create a linear regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)
# Step 5: Make predictions
y_pred = model.predict(X_test)
# Visualize the regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: Speed vs. Stopping Distance')
plt.xlabel('Speed')
plt.ylabel('Stopping Distance')
plt.legend()
plt.show()
# Step 6: Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# # Optionally, you can predict the stopping distance for a new speed
# new_speed = 60  	# predict stopping distance for a different speed
# predicted_sdistance = model.predict(np.array([[new_speed]])
# print(f"Predicted Stopping Distance for {new_speed} mph: {predicted_stopping_distance[0]}")
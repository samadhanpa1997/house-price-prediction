House Price Prediction using Linear Regression
Project Overview
This project uses a Machine Learning Linear Regression model to predict house prices based on various house features such as:

Area (Square Feet)
Number of Bedrooms
Number of Bathrooms
Age of House
The model is trained using Scikit-Learn and saved as a pickle file for future predictions.

Technologies Used
Python 3.x
Pandas
NumPy
Scikit-Learn
Matplotlib
Pickle
Project Structure
house-price-prediction/
│
├── house_data.csv
├── house_price_prediction.py
├── house_price_prediction.pkl
├── README.md
└── requirements.txt
Dataset Features
Feature	Description
area	House area in square feet
bedrooms	Number of bedrooms
bathrooms	Number of bathrooms
age	Age of the house
price	Target variable (House Price)
Data Preprocessing
The following preprocessing steps are performed:

Check for null values
Check for duplicate records
Exploratory Data Analysis (EDA)
Feature selection
Train-Test Split
Exploratory Data Analysis
Scatter Plot:

Area vs House Price
This helps understand the relationship between area and house price.

Model Training
Algorithm Used:

Linear Regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
Train-Test Split
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
Training Data: 80%
Testing Data: 20%
Model Evaluation
Evaluation Metrics:

R² Score
Mean Absolute Error (MAE)
from sklearn.metrics import r2_score, mean_absolute_error
Sample Prediction
new_house = pd.DataFrame({
    "area": [1500],
    "bedrooms": [3],
    "bathrooms": [2],
    "age": [40]
})

predicted_price = model.predict(new_house)
Saving the Model
import pickle

with open("house_price_prediction.pkl", "wb") as file:
    pickle.dump(model, file)
Visualization
Area vs House Price
Scatter plot showing relationship between house area and selling price.

Actual vs Predicted Price
Scatter plot comparing actual house prices with model predictions.

How to Run the Project
Clone Repository
git clone https://github.com/samadhanpa1997/house-price-prediction.git
Navigate to Project Folder
cd house-price-prediction
Install Required Packages
pip install -r requirements.txt
Run Application
python house_price_prediction.py
Future Enhancements
Add more house features
Hyperparameter tuning
Use advanced regression algorithms
Build REST API using FastAPI
Create UI using Streamlit
Author
Samadhan Patil

Senior Software Developer | Machine Learning Enthusiast

GitHub: https://github.com/samadhanpa1997

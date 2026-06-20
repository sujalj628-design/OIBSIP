# Import python Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Loading The Dataset
data = pd.read_csv(r"C:\Users\sujal\OneDrive\Desktop\my_project\OIBSIP\SUJALJAISWAL_TASK03\car data.csv")

print("++++++ CAR PRICE PREDICTION ++++++\n")

# DATASET EXPLORATION
print("Dataset Shape:", data.shape)

print("\nColumns:")
print(data.columns)

# CONVERTING TEXT COLUNM INTO NUMBER
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
data['Selling_type'] = data['Selling_type'].map({'Dealer': 0, 'Individual': 1})
data['Transmission'] = data['Transmission'].map({'Manual': 0, 'Automatic': 1})

# FEARTURE SELECTION
X = data[['Present_Price', 'Driven_kms', 'Fuel_Type',
            'Selling_type', 'Transmission', 'Owner']]

y = data['Selling_Price']

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# TRANING MODELAND PERFORMANCE
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

score = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE \n")
print("R2 Score:", round(score, 2))
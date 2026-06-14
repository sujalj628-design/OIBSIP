# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Loading The Dataset
data = pd.read_csv(r"C:\Users\sujal\OneDrive\Desktop\my_project\OIBSIP\SUJALJAISWAL_TASK02\Advertising.csv")

print("++++++ SALES PREDICTION USING PYTHON ++++++\n")

# DATASET Exploration AND FEATURE SELECTION
print("Dataset Shape:", data.shape)

print("\nCOLUMNS IN DATASET:")
print(data.columns)

# DATA PREPARATION AND SPLITING FOR TARANING
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split( X, 
                                                     y, 
                                                     test_size=0.2, 
                                                     random_state=42)

# MODLE
model = LinearRegression()
model.fit(X_train, y_train)

# Model Performance
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

print("\nPERFORMANCE OF THE MODLE:")
print("R2 Score:", round(score, 2))

# USER PREDICTION 
print("\nSALES PREDICTION \n")

tv = float(input("ENTER TV ADVERTISING BUDGET: "))
radio = float(input("ENTER RADIO ADVERTISING BUDGET: "))
newspaper = float(input("ENTER NEWSPAPER ADVERTISING BUDGET: "))

prediction = model.predict([[tv, radio, newspaper]])
print("\nSALES PREDICTED IS:", round(prediction[0], 2))
# Import Python Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Loading The Dataset
data = pd.read_csv(r"C:\Users\sujal\OneDrive\Desktop\my_project\OIBSIP\SUJALJAISWAL_TASK01\Iris.csv")

print("++++++ IRIS FLOWER CLASSIFICATION ++++++\n")

print("Dataset Shape:", data.shape)
print("\nSpecies Distribution:\n")
print(data['Species'].value_counts())

# Data Preparation And Spliting for traniing
x= data[['SepalLengthCm',
         'SepalWidthCm',
         'PetalLengthCm',
         'PetalWidthCm']]

y= data['Species']

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.2, 
                                                    random_state=40)

# Modle
model = RandomForestClassifier(random_state=42)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\n")
accuracy = accuracy_score(y_test, y_pred)
print("Model Performance Accuracy Score:", accuracy)

print("FLOWER PREDICTION \n")

sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))

user_data = [[sepal_length,
              sepal_width,
              petal_length, 
              petal_width]]

prediction = model.predict(user_data)
print("\nPredicted Species:", prediction[0])

# OIBSIP : DATA SCIENCE INTERNSHIP
## TASK03 : CAR PRICE PREDICTION


This project predicts the selling price of a used car using Machine Learning. A Linear Regression model is trained on historical car data to estimate the selling price based on various vehicle features.

The objective of this project is to predict the selling price of a car using its features such as present price, kilometers driven, fuel type, transmission type, ownership details, and car age.

The dataset contains information about used cars and their selling prices.

Input Features=
* Present Price
* Driven Kilometers
* Fuel Type
* Selling Type
* Transmission
* Owner
* Car Age

 
 Target Variable
* Selling Price

Technologies Used
* Python
* Pandas
* Scikit-learn

Algorithm usedin the model
* Linear Regression

Project Workflow
1. Load the dataset
2. Explore the dataset
3. Convert categorical data into numerical values
4. Create the Car Age feature
5. Select input features and target variable
6. Split the dataset into training and testing sets
7. Train the Linear Regression model
8. Evaluate the model using the R² Score
9. Predict the selling price of a car

Model Performance
**R² Score:** **0.77**

The model explains approximately **77% of the variation** in car selling prices based on the selected features.

The project allows users to enter:
* Present Price
* Driven Kilometers
* Fuel Type
* Selling Type
* Transmission
* Owner
* Car Age

The trained model predicts the estimated selling price of the car.

Conclusion-
This project demonstrates how Machine Learning can be used to estimate car selling prices. By applying the Linear Regression algorithm, the model learns the relationship between different vehicle attributes and predicts the expected selling price with good accuracy.

# traffic_prediction
DataSet Link:https://drive.google.com/file/d/1Ffg4qJIPIsJGh9qrkgMLC4jwtTOcS8xs/view?usp=sharing
The Python script trains and evaluates classification models (Decision Tree, Random Forest, Logistic Regression, SVM, MLP) to predict traffic volume at different junctions. It calculates mean absolute error (MAE) and visualizes true versus predicted values for each model and junction.

# Code Description

The provided Python script is designed to address the task of predicting traffic volume at various junctions using multiple classification models. The script begins by importing necessary libraries such as numpy, pandas, matplotlib, seaborn, and scikit-learn. It then reads traffic data from a CSV file, performs data preprocessing, and splits the data into training and testing sets.
# Model Training and Evaluation
The script proceeds to train and evaluate several classification models, including Decision Tree, Random Forest, Logistic Regression, Support Vector Machine, and Multi-Layer Perceptron (MLP). For each model and junction, it calculates the mean absolute error (MAE) and visualizes the true versus predicted values using matplotlib. The code also demonstrates the use of label encoding and feature engineering to prepare the data for model training.
# Proposed Improvements
To enhance the script, it is recommended to move the data splitting process outside the loop for consistent training and testing data. Additionally, defining the MLP model outside the loop can improve code efficiency. Furthermore, incorporating subplots for each junction and model can provide a clearer comparison of the true and predicted values. Finally, considering additional performance metrics such as accuracy, precision, and recall could offer a more comprehensive evaluation of the models.

Overall, the script serves as a foundation for training and evaluating classification models for traffic volume prediction, with potential for further refinement and expansion.

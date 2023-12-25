from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Read the DataSet
DataSets = "E:/All Courses/Fall-2023\Data Mining and Machine Learning/traffic_prediction/TrafficPredictionImprovemnet/traffic.csv"
data = pd.read_csv(DataSets)
# print the Dataset
data.head()

data = data.drop(["ID"], axis=1)
data.head()

data.info()
data.isnull().sum()

data["DateTime"] = pd.to_datetime(data["DateTime"])

data['Year'] = data['DateTime'].dt.year
data['Month'] = data['DateTime'].dt.month
data['Date_no'] = data['DateTime'].dt.day
data['Hour'] = data['DateTime'].dt.hour
data['Day'] = data['DateTime'].dt.strftime("%A")

data.head()

sns.set(style="whitegrid")

colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

plt.figure(figsize=(12, 5))

count = sns.countplot(data, x=data["Year"], hue="Junction", palette=colors)
count.set_title("Count Of Traffic On Junctions Over Years", fontsize=16)
count.set_ylabel("Number of Vehicles", fontsize=12)
count.set_xlabel("Date", fontsize=12)

plt.show()

data['Day_number'] = LabelEncoder().fit_transform(data.Day)
data.head()

correlation_matrix = data.corr()

plt.figure(figsize=(10, 8))

sns.heatmap(correlation_matrix, annot=True,
            cmap="coolwarm", fmt=".2f", linewidths=.5)

plt.title("Correlation Heatmap")
plt.show()

x = data[['Year', 'Month', 'Date_no', 'Hour', 'Day_number']]
y = data['Vehicles']

x
y

junctions = data["Junction"].unique()


bin_width = 10
data['Vehicles Binned'] = pd.cut(data['Vehicles'], bins=np.arange(
    0, data['Vehicles'].max() + bin_width, bin_width), right=False)

mlp_params = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'max_iter': 200
}

classification_models = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('Logistic Regression', LogisticRegression(solver='liblinear')),
    ('Support Vector Machine', SVC()),
    ('MLP', MLPClassifier(**mlp_params))
]

mae_values = {}

for model_name, model in classification_models:
    mae_values[model_name] = []
    for junction in junctions:
        train_size = int(0.8 * len(data))
        train_data, test_data = data[:train_size], data[train_size:]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=43, test_size=0.2)
        if model_name == 'MLP':
            model = MLPClassifier(**mlp_params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
        else:
            reg = model.fit(x_train, y_train)
            y_pred = reg.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        mae_values[model_name].append(mae)

        plt.figure(figsize=(12, 6))
        plt.plot(test_data['DateTime'], y_test,
                 label='True Values', color='blue')
        plt.plot(test_data['DateTime'], y_pred,
                 label='Predicted Values', color='red')
        plt.title(f'Junction {
                  junction} - {model_name} True vs. Predicted Vehicles (Testing Data)\nMAE: {mae:.4f}')
        plt.xlabel('Data Time)')
        plt.ylabel('Number of Vehicles')
        plt.legend()
        plt.show()

plt.figure(figsize=(12, 6))
model_names = list(mae_values.keys())
mae_scores = [np.mean(values) for values in mae_values.values()]
plt.bar(model_names, mae_scores, color='skyblue')
plt.xlabel('Classification Models')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Comparison of MAE for Different Classification Models')


for i, mae in enumerate(mae_scores):
    plt.text(i, mae, f'{mae:.4f}', ha='center', va='bottom')

plt.show()

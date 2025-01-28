import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import mlflow
from datetime import datetime

# Função para carregar os dados
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Função para preparar os dados
def prepare_data(data):
    data.dropna(inplace=True)
    X = data.drop(columns=['Potability'])
    y = data['Potability']
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Função para treinar e avaliar os modelos
def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        recall = recall_score(y_test, y_pred, zero_division=1)
        f1 = f1_score(y_test, y_pred, zero_division=1)

        report = classification_report(y_test, y_pred, output_dict=True)
        results.append((model_name, model, accuracy, precision, recall, f1, report))
    return results

# Função para registrar os resultados no MLflow
def log_results(results):
    mlflow.set_tracking_uri('http://localhost:5001/')
    mlflow.set_experiment('Water Quality Prediction')
    now = datetime.now()
    current_time = now.strftime("%d/%m/%Y - %H:%M:%S")
    
    for model_name, model, accuracy, precision, recall, f1, report in results:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_dict(report, "classification_report.json")
            mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.set_tag("data", current_time)

# Função principal para executar o pipeline
def main():
    sample_url = "water_potability.csv"
    data = load_data(sample_url)
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True),
        'Neural Network': MLPClassifier(max_iter=500)
    }
    
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    log_results(results)

if __name__ == "__main__":
    main()
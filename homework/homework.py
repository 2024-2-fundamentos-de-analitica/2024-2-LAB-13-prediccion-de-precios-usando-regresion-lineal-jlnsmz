import os
import gzip
import json
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score


# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
def limpiarDatos(df: pd.DataFrame):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df = df.drop(columns=["Year", "Car_Name"])
    x, y = df.drop(columns=["Present_Price"]), df["Present_Price"]
    return df, x, y


# Paso 2.
# Cree un pipeline para el modelo de regresión.
def pipeline() -> Pipeline:
    caracteristicas = ["Fuel_Type", "Selling_type", "Transmission"]
    caracteristicasNum = ["Selling_Price", "Driven_kms", "Owner", "Age"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), caracteristicas),
            ('scaler', MinMaxScaler(), caracteristicasNum),
        ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ('feature_selection', SelectKBest(score_func=f_regression)),
        ('regressor', LinearRegression()),
    ])
    return pipeline


# Paso 3.
# Optimice los hiperparámetros del pipeline usando validación cruzada.
def hiperParametros(pipeline, x, y):
    parametros = {
        "feature_selection__k": range(1, 12),
    }
    gridSearch = GridSearchCV(
        pipeline,
        parametros,
        cv=10,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    return gridSearch.fit(x, y)


# Paso 4.
# Guarde el modelo comprimido.
def guardar(model):
    os.makedirs('files/models', exist_ok=True)
    with gzip.open('files/models/model.pkl.gz', 'wb') as file:
        pickle.dump(model, file)


# Paso 5.
# Calcule las métricas y guárdelas en un archivo JSON.
def metricas(pipeline, x_train, y_train, x_test, y_test):
    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)

    metricasTrain = {
        "type": "metrics",
        "dataset": "train",
        "r2": float(r2_score(y_train, y_train_pred)),
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "mad": float(median_absolute_error(y_train, y_train_pred)),
    }

    metricasTest = {
        "type": "metrics",
        "dataset": "test",
        "r2": float(r2_score(y_test, y_test_pred)),
        "mse": float(mean_squared_error(y_test, y_test_pred)),
        "mad": float(median_absolute_error(y_test, y_test_pred)),
    }

    return metricasTrain, metricasTest


def guardarMetricas(metrics_train, metrics_test, file_path="files/output/metrics.json"):
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    metricas = [metrics_train, metrics_test]

    with open(file_path, "w") as f:
        for i in metricas:
            f.write(json.dumps(i) + "\n")


# Cargar datos
test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")

# Preprocesar datos
test, x_test, y_test = limpiarDatos(test)
train, x_train, y_train = limpiarDatos(train)

# Entrenar modelo
modelo = pipeline()
modelo = hiperParametros(modelo, x_train, y_train)
guardar(modelo)

# Calcular métricas
metrics_train, metrics_test = metricas(modelo, x_train, y_train, x_test, y_test)
guardarMetricas(metrics_train, metrics_test)
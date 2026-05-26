from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import Home as h
import streamlit as st


X_train_rf = h.X_train.reshape(h.X_train.shape[0], h.X_train.shape[1]*h.X_train.shape[2])

X_test_rf = h.X_test.reshape(h.X_test.shape[0],h.X_test.shape[1]*h.X_test.shape[2])

rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)

rf_model.fit(X_train_rf, h.y_train)

joblib.dump(rf_model, "nepse-data/src/RandomForest.pkl")

predictions = rf_model.predict(X_test_rf)

mse = mean_squared_error(h.y_test, predictions)

print("RF MSE:",mse)
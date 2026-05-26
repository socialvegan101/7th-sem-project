from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import Home as h
import streamlit as st
import os

# Flatten sequence

X_train_svm = h.X_train.reshape(h.X_train.shape[0], h.X_train.shape[1]*h.X_train.shape[2])

X_test_svm = h.X_test.reshape(h.X_test.shape[0], h.X_test.shape[1]*h.X_test.shape[2])

# Create model
svm_model = SVR(kernel='rbf', C=100, gamma=0.1)

# Train
svm_model.fit( X_train_svm, h.y_train)


# Save model
joblib.dump( svm_model, "nepse-data/src/SVM_model.pkl")

# Prediction
predictions = svm_model.predict(X_test_svm)
mse = mean_squared_error(h.y_test, predictions)  
rmse = np.sqrt(mse)
print("SVM MSE:",mse)
print("RMSE:",rmse )

           
            
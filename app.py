import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# PAGE CONFIG
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")

st.title("Iris Flower Classification App")
st.write("Support Vector Machine (SVM) Classifier")

# LOAD DATASET
iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

# SHOW DATASET
st.subheader("Dataset")
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

if st.checkbox("Show Dataset"):
    st.write(df.head())

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# FEATURE SCALING
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL TRAINING
model = SVC(kernel="linear")

model.fit(X_train, y_train)

# MODEL EVALUATION
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")

st.success(f"Accuracy: {accuracy:.2f}")

# CONFUSION MATRIX
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

ax.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm[0])):
        ax.text(j, i, cm[i, j], ha="center", va="center")

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# USER INPUT
st.subheader("Predict Flower Type")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)

sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)

petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)

petal_width = st.slider("Petal Width", 0.1, 3.0, 1.0)

# PREDICTION
if st.button("Predict Flower"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    flower_name = target_names[prediction[0]]

    st.success(f"Predicted Flower: {flower_name}")

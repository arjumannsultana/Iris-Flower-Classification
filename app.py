import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# -----------------------------
# Cherry Blossom Animation
# -----------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(to bottom, #ffe6f0, #ffffff);
}

@keyframes fall {
  0% { transform: translateY(-10vh) rotate(0deg); }
  100% { transform: translateY(110vh) rotate(360deg); }
}

.sakura {
  position: fixed;
  top: -10px;
  font-size: 24px;
  animation: fall linear infinite;
  opacity: 0.8;
}

</style>

<div class="sakura" style="left:10%; animation-duration:10s;">🌸</div>
<div class="sakura" style="left:20%; animation-duration:12s;">🌸</div>
<div class="sakura" style="left:30%; animation-duration:9s;">🌸</div>
<div class="sakura" style="left:40%; animation-duration:11s;">🌸</div>
<div class="sakura" style="left:50%; animation-duration:13s;">🌸</div>
<div class="sakura" style="left:60%; animation-duration:10s;">🌸</div>
<div class="sakura" style="left:70%; animation-duration:12s;">🌸</div>
<div class="sakura" style="left:80%; animation-duration:9s;">🌸</div>
<div class="sakura" style="left:90%; animation-duration:14s;">🌸</div>

""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("iris_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# Flower Images
# -----------------------------
flower_images = {
    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa.jpg",
    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# -----------------------------
# Header
# -----------------------------
st.title("🌸 Iris Flower Classification")
st.markdown("Predict the **species of Iris flower** using Machine Learning.")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🌼 Input Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 0.0, 10.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 0.0, 10.0, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 0.0, 10.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.0, 10.0, 0.2)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("🔍 Predict Species"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    species = encoder.inverse_transform(prediction)
    confidence = np.max(probabilities) * 100

    st.success(f"🌺 Predicted Species: **{species[0]}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}%**")

    # Show Flower Image
    if species[0] in flower_images:
        st.image(
            flower_images[species[0]],
            caption=species[0],
            width=250
        )

# -----------------------------
# Feature Importance
# -----------------------------
st.markdown("### 🔎 Feature Importance")

feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
importances = model.feature_importances_

for feature, importance in zip(feature_names, importances):
    st.write(f"{feature}: {importance:.3f}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "📌 **Project by Arjuman Sultana**  \n"
    "Machine Learning | Streamlit | Python"
)
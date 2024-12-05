import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Judul aplikasi
st.title("Aplikasi Klasifikasi dengan SVM dan Random Forest")


# Upload dataset
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset:")
    st.dataframe(data)
    
    # Memilih fitur dan label
    features = st.multiselect("Pilih fitur (kolom input)", data.columns.tolist())
    label = st.selectbox("Pilih label (kolom output)", data.columns.tolist())
    
    if features and label:
        X = data[features]
        y = data[label]
        
        # Menangani label kontinu
        if y.dtype in [np.float64, np.int64] and len(np.unique(y)) > 10:
            st.warning("Label terdeteksi sebagai nilai kontinu. Akan dilakukan diskretisasi.")
            bins = st.slider("Tentukan jumlah kategori untuk label", 2, 10, 3)  # Default 3 kategori
            y = pd.cut(y, bins=bins, labels=[f"Class {i}" for i in range(bins)])
            st.write("Label dikonversi ke kategori:")
            st.dataframe(pd.DataFrame({label: y}))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Memilih model klasifikasi
        classifier_name = st.selectbox("Pilih model klasifikasi", ["SVM", "Random Forest"])
        
        if classifier_name == "SVM":
            # Hyperparameter SVM
            kernel = st.selectbox("Pilih kernel SVM", ["linear", "poly", "rbf", "sigmoid"])
            model = SVC(kernel=kernel)
        elif classifier_name == "Random Forest":
            # Hyperparameter Random Forest
            n_estimators = st.slider("Jumlah pohon (n_estimators)", 10, 500, 100, step=10)
            max_depth = st.slider("Kedalaman maksimum pohon (max_depth)", 1, 50, 10, step=1)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        # Training model
        model.fit(X_train, y_train)
        
        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        st.pyplot(fig)
        
        # Prediksi data baru
        st.subheader("Prediksi Data Baru")
        input_data = {}
        for feature in features:
            input_data[feature] = st.number_input(f"Masukkan nilai untuk {feature}")
        
        if st.button("Prediksi"):
            input_array = np.array([list(input_data.values())])
            prediction = model.predict(input_array)
            st.write(f"Prediksi: {prediction[0]}")
else:
    st.write("Silakan upload dataset untuk memulai.")

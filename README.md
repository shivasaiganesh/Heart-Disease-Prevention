# ❤️ Heart Disease Risk Predictor

![Heart Health Dashboard](https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80)

This is a Support Vector Machine (SVM) based machine learning model deployed via a **Streamlit** Python web application. It predicts the probability of heart disease using a patient's medical features.

## 🚀 Features
- **Interactive Web Interface:** Built using Streamlit, allowing users to easily input medical data.
- **Machine Learning Model:** Uses an SVM classifier trained on the Kaggle Heart Disease dataset, utilizing GridSearchCV for optimal parameter selection.
- **Risk Assessment:** Provides a clear indication of Heart Disease Risk (Low, Medium, High) along with a probability score.
- **Data Preprocessing:** Handles missing values and scales numeric features automatically using Scikit-Learn pipelines.

## 📸 Demo Screenshots

![Medical Tech](https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80) 

*(Replace these example images with actual screenshots of your application)*

## 🛠️ Tech Stack
- **Python** 3.x
- **Streamlit** (Web Framework)
- **Scikit-Learn** (Machine Learning Pipeline and SVM model)
- **Pandas** & **NumPy** (Data Manipulation)
- **Joblib** (Model Serialization)

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shivasaiganesh/Heart-Disease-Prevention.git
   cd Heart-Disease-Prevention
   ```

2. **Install the dependencies:**
   ```bash
   pip install streamlit pandas joblib scikit-learn
   ```

3. **Run the Model Training (Optional):**
   *(If you want to retrain the SVM model)*
   ```bash
   python train_model.py
   ```

4. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## ⚠️ Disclaimer
This application is a demonstration for coursework and research purposes only. **It must NOT be used for real medical diagnosis or treatment.**
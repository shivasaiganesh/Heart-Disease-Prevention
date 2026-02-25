# Heart Disease Prevention

Simple Flask app and trained SVM model to predict heart disease risk from clinical features.

Getting started

- Requirements: Python 3.8+ and common data science packages (pandas, scikit-learn, Flask).
- Run the web app:

```powershell
cd heartdisease_codes
python app.py
```

Files
- `app.py`: Flask application to serve predictions.
- `train_model.py`: training script used to create `heart_svm_model.pkl`.
- `heart_svm_model.pkl`: trained SVM model.
- `heart.csv`: sample dataset.

Notes
- Remove or avoid committing large binary model files if you prefer using Git LFS.

License
- Add a license file if you want to make this repo public.
# Heart-Disease-Prevention
# GlycoSense — Diabetes Prediction Web App

Final Year Project: Diabetes Progression Prediction Using Regression & Ensemble Models

---

## SETUP (3 steps)

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run the app
```
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## Project Structure

```
diabetes_app/
├── app.py                  ← Flask backend (model training + API)
├── requirements.txt        ← Python dependencies
├── README.md
└── templates/
    └── index.html          ← Frontend UI
```

---

## How It Works

1. On startup, `app.py` downloads the Pima dataset, cleans it, engineers your 4 features,
   trains your GradientBoostingClassifier, and initialises the SHAP explainer.

2. The user fills in 8 raw features on the UI.

3. POST /api/predict → derives 4 engineered features → scales → predicts → computes SHAP values.

4. The frontend renders:
   - Risk score ring (% probability of diabetes)
   - SHAP bar chart (which features drove the prediction)
   - Personalised lifestyle recommendations

---

## Model Pipeline (matches your notebook exactly)

- Dataset: Pima Indians Diabetes (768 samples, 8 features)
- Cleaning: Zero-replacement with column mean for Glucose, BP, Skin, Insulin, BMI
- Feature Engineering:
  - GIR = Glucose / Insulin
  - BMI_Age_Interaction = BMI × Age
  - BP_BMI_Interaction = BloodPressure × BMI
  - Is_Obese = 1 if BMI > 30
- Scaling: StandardScaler (fit on train, transform on test + user input)
- Model: GradientBoostingClassifier(n_estimators=100, random_state=42)
- Explainability: SHAP TreeExplainer

---

## Requirements

- Python 3.8+
- Internet connection on first run (to download Pima dataset from GitHub)

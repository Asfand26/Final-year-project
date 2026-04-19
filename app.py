"""
GlycoSense — Diabetes Prediction Web App
=========================================
Backend: Flask + your exact GradientBoostingClassifier pipeline
Run:  python app.py
Open: http://localhost:5000
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import shap
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ── GLOBAL MODEL STATE ───────────────────────────────────────────────────────
model_state = {}

def train_model():
    """Replicates your notebook pipeline exactly."""
    print("\n🔬 GlycoSense — Loading and training model...")

    # 1. Load Pima dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    col_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=col_names)
    print("   ✓ Dataset loaded (768 samples)")

    # 2. Clean zero values
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_replace:
        df[col].replace(0, np.nan, inplace=True)
        df[col].fillna(df[col].mean(), inplace=True)

    # 3. Feature engineering (your 4 unique features)
    df['GIR'] = df['Glucose'] / df['Insulin']
    df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
    df['BP_BMI_Interaction'] = df['BloodPressure'] * df['BMI']
    df['Is_Obese'] = (df['BMI'] > 30).astype(int)
    print("   ✓ 4 engineered features created (GIR, BMI×Age, BP×BMI, Is_Obese)")

    # 4. Split and scale
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

    # 5. Train GradientBoostingClassifier (your model)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_scaled_df, y_train)
    predictions = gb.predict(X_test_scaled_df)
    accuracy = accuracy_score(y_test, predictions)
    print(f"   ✓ Model trained — Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

    # 6. SHAP explainer
    explainer = shap.TreeExplainer(gb)
    print("   ✓ SHAP explainer initialised")

    # 7. Store report for display
    report = classification_report(y_test, predictions,
                                   target_names=['No Diabetes', 'Diabetes'],
                                   output_dict=True)

    model_state.update({
        'gb': gb,
        'scaler': scaler,
        'explainer': explainer,
        'feature_cols': list(X_train_scaled_df.columns),
        'accuracy': round(accuracy * 100, 2),
        'report': report
    })
    print("✅ GlycoSense is ready!\n")


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', accuracy=model_state['accuracy'])


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Parse the 8 raw inputs
        pregnancies = float(data['pregnancies'])
        glucose     = float(data['glucose'])
        bp          = float(data['bloodPressure'])
        skin        = float(data['skinThickness'])
        insulin     = float(data['insulin'])
        bmi         = float(data['bmi'])
        dpf         = float(data['dpf'])
        age         = float(data['age'])

        # Guard against division by zero for GIR
        if insulin == 0:
            insulin = 1.0

        # Build user DataFrame
        user_dict = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bp,
            'SkinThickness': skin,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        user_df = pd.DataFrame([user_dict])

        # Feature engineering — MUST match training pipeline
        user_df['GIR'] = user_df['Glucose'] / user_df['Insulin']
        user_df['BMI_Age_Interaction'] = user_df['BMI'] * user_df['Age']
        user_df['BP_BMI_Interaction'] = user_df['BloodPressure'] * user_df['BMI']
        user_df['Is_Obese'] = (user_df['BMI'] > 30).astype(int)

        # Align column order and scale
        user_df = user_df[model_state['feature_cols']]
        user_scaled = model_state['scaler'].transform(user_df)
        user_scaled_df = pd.DataFrame(user_scaled, columns=model_state['feature_cols'])

        # Predict
        prediction = int(model_state['gb'].predict(user_scaled_df)[0])
        probas     = model_state['gb'].predict_proba(user_scaled_df)[0]
        risk_score = round(float(probas[1]) * 100, 2)

        # SHAP values
        shap_values = model_state['explainer'].shap_values(user_scaled_df)
        # For binary classification TreeExplainer returns array of shape (1, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]   # class-1 SHAP values
        else:
            sv = shap_values[0]

        shap_data = []
        for feat, val in zip(model_state['feature_cols'], sv):
            shap_data.append({
                'feature': feat,
                'shap_value': round(float(val), 4),
                'raw_value': round(float(user_df.iloc[0][feat]), 4)
            })
        # Sort by absolute SHAP value descending
        shap_data.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        # Lifestyle recommendations based on inputs + prediction
        recommendations = generate_recommendations(
            prediction, risk_score, glucose, bmi, bp, insulin, dpf, age, pregnancies
        )

        return jsonify({
            'success': True,
            'prediction': prediction,
            'risk_score': risk_score,
            'confidence': round(float(max(probas)) * 100, 1),
            'shap': shap_data,
            'recommendations': recommendations,
            'model_accuracy': model_state['accuracy']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


def generate_recommendations(prediction, risk_score, glucose, bmi, bp,
                               insulin, dpf, age, pregnancies):
    """Generate personalised xAI-driven lifestyle recommendations."""
    recs = []

    # Glucose
    if glucose > 140:
        recs.append({
            'icon': '🍽️',
            'title': 'Reduce Refined Carbohydrates',
            'desc': f'Your glucose level ({glucose:.0f} mg/dL) is elevated. Limit white rice, bread, and sugary drinks. Opt for low-GI foods like oats, lentils, and leafy greens.',
            'priority': 'high'
        })
    elif glucose > 110:
        recs.append({
            'icon': '🍽️',
            'title': 'Monitor Carbohydrate Intake',
            'desc': f'Glucose at {glucose:.0f} mg/dL is borderline. Consider portion control and replacing processed carbs with whole grains and fibre-rich foods.',
            'priority': 'medium'
        })

    # BMI
    if bmi > 35:
        recs.append({
            'icon': '🏃',
            'title': 'Structured Weight Management',
            'desc': f'BMI of {bmi:.1f} indicates obesity. Aim for 150+ min/week of aerobic exercise. Even a 5–7% weight reduction significantly lowers diabetes risk.',
            'priority': 'high'
        })
    elif bmi > 30:
        recs.append({
            'icon': '🚶',
            'title': 'Increase Physical Activity',
            'desc': f'BMI of {bmi:.1f} puts you in the obese range. Start with 30-minute daily walks and gradually add resistance training 2–3×/week.',
            'priority': 'high'
        })
    elif bmi > 25:
        recs.append({
            'icon': '🧘',
            'title': 'Maintain Healthy Weight',
            'desc': f'BMI of {bmi:.1f} is slightly above ideal. Light activity like yoga, cycling, or swimming 4×/week can help prevent further increase.',
            'priority': 'medium'
        })

    # Blood Pressure
    if bp > 90:
        recs.append({
            'icon': '🫀',
            'title': 'Manage Blood Pressure',
            'desc': f'Diastolic BP of {bp:.0f} mmHg is elevated. Reduce sodium intake, avoid processed foods, and practice stress-reduction techniques like meditation.',
            'priority': 'high'
        })
    elif bp > 80:
        recs.append({
            'icon': '🫀',
            'title': 'Monitor Blood Pressure',
            'desc': f'BP ({bp:.0f} mmHg) is at the upper edge of normal. Regular monitoring and a DASH diet (low sodium, high potassium) is recommended.',
            'priority': 'medium'
        })

    # Insulin resistance (GIR)
    gir = glucose / max(insulin, 1)
    if gir > 1.5:
        recs.append({
            'icon': '💉',
            'title': 'Address Insulin Resistance',
            'desc': f'Your Glucose-to-Insulin Ratio ({gir:.2f}) suggests insulin resistance. Intermittent fasting, reduced sugar intake, and strength training can improve sensitivity.',
            'priority': 'high'
        })

    # Family history (DPF)
    if dpf > 0.8:
        recs.append({
            'icon': '🧬',
            'title': 'Monitor Genetic Risk Factors',
            'desc': f'Your Diabetes Pedigree Function ({dpf:.2f}) indicates significant family history. Schedule HbA1c tests every 6 months and discuss preventive care with your physician.',
            'priority': 'high'
        })
    elif dpf > 0.4:
        recs.append({
            'icon': '🧬',
            'title': 'Stay Aware of Family History',
            'desc': f'Moderate family risk (DPF: {dpf:.2f}). Annual diabetes screening is advisable alongside a healthy lifestyle.',
            'priority': 'medium'
        })

    # Age-specific
    if age > 45:
        recs.append({
            'icon': '📅',
            'title': 'Age-Related Screening',
            'desc': 'Adults over 45 should have fasting blood glucose checked annually. Early detection is the most effective prevention strategy.',
            'priority': 'medium'
        })

    # Sleep and stress (always relevant)
    recs.append({
        'icon': '😴',
        'title': 'Prioritise Sleep Quality',
        'desc': 'Poor sleep (< 6 hrs) disrupts insulin sensitivity and increases cortisol, which raises blood sugar. Target 7–9 hours of consistent, quality sleep.',
        'priority': 'low'
    })

    recs.append({
        'icon': '💧',
        'title': 'Stay Hydrated',
        'desc': 'Adequate water intake (2–3 L/day) helps kidneys flush excess glucose. Replace sugary drinks and juices with water or unsweetened herbal teas.',
        'priority': 'low'
    })

    # Low risk bonus
    if prediction == 0 and risk_score < 30:
        recs.insert(0, {
            'icon': '✅',
            'title': 'Maintain Your Healthy Lifestyle',
            'desc': f'Your risk score of {risk_score:.1f}% is low. Keep up regular exercise, balanced nutrition, and annual check-ups to stay protected.',
            'priority': 'low'
        })

    return recs[:7]   # Cap at 7 cards


@app.route('/api/metrics')
def metrics():
    return jsonify({
        'accuracy': model_state['accuracy'],
        'report': model_state['report'],
        'features': model_state['feature_cols']
    })


# ── STARTUP ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    train_model()
    print("🌐 Open your browser at: http://localhost:5000\n")
    app.run(debug=True, port=5000)

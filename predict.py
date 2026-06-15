import pickle
import numpy as np
import pandas as pd
import shap

MODEL_PATH = 'model/universal_model.pkl'

CATEGORICAL_COLS = [
    'Gender', 'MaritalStatus', 'Department',
    'JobRole', 'OverTime', 'BusinessTravel'
]

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def preprocess_input(df, model_data):
    df = df.copy()
    encoders = model_data['encoders']
    features = model_data['features']

    # Fill missing
    for col in df.columns:
        try:
            if df[col].dtype == object or str(df[col].dtype) == 'string':
                df[col] = df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                )
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
        except:
            df[col] = df[col].fillna('Unknown')

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns and col in encoders:
            le = encoders[col]
            df[col] = df[col].apply(
                lambda x: le.transform([str(x)])[0]
                if str(x) in le.classes_ else 0
            )

    # Add missing features as 0
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

def predict_single(input_dict, model_data):
    model = model_data['model']
    features = model_data['features']
    df = pd.DataFrame([input_dict])
    df = preprocess_input(df, model_data)
    prob = model.predict_proba(df)[0][1]
    label = get_risk_label(prob)
    factors = get_shap_factors(model, df, features)
    return prob, label, factors

def predict_bulk(df_raw, model_data):
    model = model_data['model']
    features = model_data['features']

    # Drop target if present
    df = df_raw.copy()
    for col in ['Attrition', 'attrition']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df = preprocess_input(df, model_data)
    probs = model.predict_proba(df)[:, 1]
    labels = [get_risk_label(p) for p in probs]
    return probs, labels

def get_risk_label(prob):
    if prob < 0.3:
        return "🟢 Low Risk"
    elif prob < 0.6:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"

def get_shap_factors(model, df, features, top_n=5):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)
        vals = shap_values[0]
        indices = np.argsort(np.abs(vals))[::-1][:top_n]
        factors = []
        for i in indices:
            factors.append({
                'feature': features[i],
                'impact': float(vals[i]),
                'value': float(df.iloc[0][features[i]])
            })
        return factors
    except:
        return []
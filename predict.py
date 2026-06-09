import pickle
import numpy as np
import pandas as pd
import shap

MODEL_PATH = 'model/universal_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def preprocess_input(df, model_data):
    """Preprocess any input dataframe using saved encoders"""
    df = df.copy()
    encoders = model_data['encoders']
    drop_cols = model_data['drop_cols']
    features = model_data['features']

    # Drop same cols as training
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop target if present
    target_col = model_data['target_col']
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Fill missing
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: le.transform([str(x)])[0]
                if str(x) in le.classes_ else 0
            )

    # Add missing features
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]

def predict_bulk(df_raw, model_data):
    """Predict risk for entire dataframe"""
    model = model_data['model']
    features = model_data['features']

    df = preprocess_input(df_raw, model_data)
    probs = model.predict_proba(df)[:, 1]
    labels = [get_risk_label(p) for p in probs]
    return probs, labels

def predict_single(input_dict, model_data):
    """Predict risk for single row"""
    model = model_data['model']
    features = model_data['features']

    df = pd.DataFrame([input_dict])
    df = preprocess_input(df, model_data)
    prob = model.predict_proba(df)[0][1]
    label = get_risk_label(prob)
    factors = get_shap_factors(model, df, features)
    return prob, label, factors

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
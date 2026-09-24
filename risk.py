def estimate_risk(confidence):
    if confidence > 0.85:
        return "High Risk"
    elif confidence > 0.6:
        return "Medium Risk"
    else:
        return "Low Risk"
def calculate_risk(detected_classes):

    risk_score = 0
    violations = []

    if "no_helmet" in detected_classes:
        risk_score += 40
        violations.append("Helmet Missing")

    if "no_gloves" in detected_classes:
        risk_score += 20
        violations.append("Gloves Missing")

    if "no_boots" in detected_classes:
        risk_score += 20
        violations.append("Boots Missing")

    if "no_goggle" in detected_classes:
        risk_score += 20
        violations.append("Goggles Missing")

    if "no_vest" in detected_classes:
        risk_score += 20
        violations.append("Vest Missing")

    if risk_score >= 60:
        risk_level = "HIGH RISK"
    elif risk_score >= 20:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "SAFE"

    return risk_score, risk_level, violations
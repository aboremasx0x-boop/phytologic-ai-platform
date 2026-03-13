# forecast.py
from typing import Dict, Any, List

def daily_risk(context: Dict[str, Any]) -> float:
    """
    Returns risk in [0..1] based on simple tomato foliar disease logic
    using leaf wetness + temperature + humidity.
    """
    T = float(context.get("temperature_c", 0))
    RH = float(context.get("humidity_pct", 0))
    LWH = float(context.get("leaf_wetness_hours", 0))

    risk = 0.0

    # leaf wetness is a strong driver
    if LWH >= 6: risk += 0.45
    elif LWH >= 3: risk += 0.25

    # temp window (roughly favorable moderate/warm)
    if 18 <= T <= 30: risk += 0.35
    elif 12 <= T < 18 or 30 < T <= 34: risk += 0.15

    # high RH supports infection periods
    if RH >= 90: risk += 0.25
    elif RH >= 75: risk += 0.15

    return max(0.0, min(1.0, risk))

def forecast_severity(severity_now: float, context: Dict[str, Any], days: int = 7) -> Dict[str, Any]:
    """
    Forecast severity curve for the next N days using logistic growth scaled by risk.
    """
    S = float(max(0.0, min(100.0, severity_now)))
    risk = daily_risk(context)

    # base growth rate (tune later from field data)
    r = 0.18  # per day

    curve: List[float] = [round(S, 2)]
    for _ in range(days):
        S = S + (r * (risk) * S * (1 - S/100.0))
        S = max(0.0, min(100.0, S))
        curve.append(round(S, 2))

    level = "low"
    if risk >= 0.7: level = "high"
    elif risk >= 0.4: level = "medium"

    return {
        "risk_score": round(risk, 2),
        "risk_level": level,
        "severity_curve_7d": curve  # includes today at index 0
    }
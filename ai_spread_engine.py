from collections import defaultdict


class DiseaseSpreadEngine:
    def __init__(self):
        pass

    def calculate_risk_score(self, temp, humidity, rain, severity, cases):
        temp_factor = self._temp_factor(temp)
        humidity_factor = humidity / 100
        rain_factor = min(rain / 10, 1)
        severity_factor = severity / 100
        cases_factor = min(cases / 20, 1)

        score = (
            temp_factor * 0.25 +
            humidity_factor * 0.2 +
            rain_factor * 0.15 +
            severity_factor * 0.25 +
            cases_factor * 0.15
        )

        return round(score * 10, 2)

    def _temp_factor(self, temp):
        if 20 <= temp <= 30:
            return 1
        elif 15 <= temp < 20 or 30 < temp <= 35:
            return 0.7
        else:
            return 0.3

    def classify_risk(self, score):
        if score < 3:
            return "منخفض"
        elif score < 6:
            return "متوسط"
        else:
            return "مرتفع"

    def generate_heatmap_points(self, diagnoses):
        points = []

        for d in diagnoses:
            if not d.get("latitude") or not d.get("longitude"):
                continue

            risk = self.calculate_risk_score(
                temp=d.get("temperature", 30),
                humidity=d.get("humidity", 50),
                rain=d.get("rainfall", 0),
                severity=d.get("severity_percent", 0),
                cases=1
            )

            points.append({
                "lat": d["latitude"],
                "lon": d["longitude"],
                "risk": risk
            })

        return points

    def regional_analysis(self, diagnoses):
        regions = defaultdict(list)

        for d in diagnoses:
            regions[d.get("region", "unknown")].append(d)

        result = []

        for region, items in regions.items():
            avg_severity = sum(d.get("severity_percent", 0) for d in items) / len(items)

            score = self.calculate_risk_score(
                temp=30,
                humidity=60,
                rain=0,
                severity=avg_severity,
                cases=len(items)
            )

            result.append({
                "region": region,
                "cases": len(items),
                "avg_severity": round(avg_severity, 2),
                "risk_score": score,
                "risk_level": self.classify_risk(score)
            })

        return result

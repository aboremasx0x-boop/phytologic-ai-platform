class DiseaseSpreadEngine:

    def calculate_risk(self, temperature, humidity, severity, quality):

        if temperature < 10:
            temp_score = 0.2
        elif temperature < 20:
            temp_score = 0.5
        elif temperature < 30:
            temp_score = 0.9
        else:
            temp_score = 0.7

        humidity_score = humidity / 100
        severity_score = severity / 100
        quality_score = quality

        risk = (
            humidity_score * 0.4 +
            temp_score * 0.2 +
            severity_score * 0.3 +
            quality_score * 0.1
        )

        return round(risk * 100, 2)

    def classify_risk(self, risk):
        if risk < 30:
            return "منخفض"
        elif risk < 60:
            return "متوسط"
        else:
            return "مرتفع"

    def recommendation(self, risk):
        if risk < 30:
            return "مراقبة فقط"
        elif risk < 60:
            return "إجراءات وقائية"
        else:
            return "تدخل فوري"

    def future_projection(self, risk):
        if risk < 30:
            return "مستقر خلال 72 ساعة"
        elif risk < 60:
            return "قد يزيد خلال 48 ساعة"
        else:
            return "انتشار سريع خلال 24 ساعة"

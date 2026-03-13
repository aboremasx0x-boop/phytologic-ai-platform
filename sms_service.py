import requests


class SMSService:
    def __init__(self, app_sid: str = "", sender: str = "Phytologic"):
        self.app_sid = app_sid
        self.sender = sender

    def is_configured(self) -> bool:
        return bool(self.app_sid and str(self.app_sid).strip())

    def send_sms(self, phone: str, message: str):
        if not self.app_sid:
            return {"status": "sms_disabled"}

        url = "https://api.unifonic.com/rest/SMS/messages"

        payload = {
            "AppSid": self.app_sid,
            "SenderID": self.sender,
            "Recipient": phone,
            "Body": message
        }

        try:
            response = requests.post(url, data=payload, timeout=20)

            return {
                "success": response.ok,
                "status_code": response.status_code,
                "response_text": response.text
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

import smtplib
import os
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='config.env')

EMAIL_ADDRESS = os.getenv('GMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('GMAIL_APP_PASSWORD')
PHONE_NUMBER = os.getenv('TARGET_PHONE_NUMBER')
CARRIER_GATEWAY = "@tmomail.net"  # Adjust if needed

def send_sms_alert(message):
    to_number = PHONE_NUMBER + CARRIER_GATEWAY
    msg = MIMEText(message)
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = to_number
    msg['Subject'] = "AI Caretaker Alert"

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            response = server.sendmail(EMAIL_ADDRESS, to_number, msg.as_string())

            # Check if there were any failed deliveries
            if response == {}:
                print(f"✅ SMS alert sent successfully to {to_number}!")
            else:
                print(f"❌ Partial Failure. Server response: {response}")
    except smtplib.SMTPRecipientsRefused:
        print(f"❌ Failed: The recipient address {to_number} was refused.")
    except smtplib.SMTPAuthenticationError:
        print("❌ Failed: Authentication error - check your Gmail login or App Password.")
    except Exception as e:
        print(f"❌ Failed: {e}")
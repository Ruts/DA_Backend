from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai
from twilio.rest import Client

genai.configure(api_key="AIzaSyBF5ys8AH4ZWeyvy1oVHj8NRIVcDA9ad_I")
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"

# Twilio credentials (use environment variables in production)
TWILIO_ACCOUNT_SID = "ACc58ae7c23b02e197cdd55b8ef5a40f7a"
TWILIO_AUTH_TOKEN = "a0bf49fc3d95bef0be042fdc02da54ad"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio sandbox number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")

    response = model.generate_content(question)
    answer = response.text

    return {"answer": answer}

@app.post("/send-whatsapp")
async def send_whatsapp(request: Request):
    data = await request.json()
    to_number = data.get("to")  # Format: whatsapp:+2547xxxxxxx
    message = data.get("message")

    try:
        sent = client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_NUMBER,
            to="whatsapp:+" + to_number
        )
        return {"status": "sent", "sid": sent.sid}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# @app.post("/receive-whatsapp")
# async def receive_whatsapp(request: Request):
#     data = await request.form()
#     from_number = data.get("From")
#     message_body = data.get("Body")

#     # You can process or respond to the message here
#     return {"from": from_number, "message": message_body}

@app.post("/receive-whatsapp")
async def receive_whatsapp(request: Request):
    try:
        data = await request.form()
        print("Received form data:", data)

        from_number = data.get("From")
        message_body = data.get("Body")

        return {"from": from_number, "message": message_body}
    except Exception as e:
        print("Error receiving WhatsApp message:", str(e))
        return {"error": str(e)}

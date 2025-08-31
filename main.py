from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai
from twilio.rest import Client
import requests
from io import BytesIO
from PIL import Image

genai.configure(api_key="AIzaSyBF5ys8AH4ZWeyvy1oVHj8NRIVcDA9ad_I")
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"
image_model = genai.GenerativeModel("gemini-2.5-flash-image-preview")  # for image input

# Twilio credentials (use environment variables in production)
TWILIO_ACCOUNT_SID = "ACc58ae7c23b02e197cdd55b8ef5a40f7a"
TWILIO_AUTH_TOKEN = "a0bf49fc3d95bef0be042fdc02da54ad"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  # Twilio sandbox number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# At the top of your FastAPI file
latest_whatsapp_message = {"from": "", "message": ""}

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

# @app.post("/receive-whatsapp")
# async def receive_whatsapp(request: Request):
#     global latest_whatsapp_message
#     data = await request.form()
#     from_number = data.get("From")
#     message_body = data.get("Body")

#     latest_whatsapp_message = {
#         "from": from_number,
#         "message": message_body
#     }

#     print("Stored incoming message:", latest_whatsapp_message)
#     return {"status": "received"}

@app.get("/latest-whatsapp")
async def get_latest_whatsapp():
    return latest_whatsapp_message

# @app.post("/receive-whatsapp")
# async def receive_whatsapp(request: Request):
#     global latest_whatsapp_message
#     data = await request.form()
#     from_number = data.get("From")  # Format: whatsapp:+2547xxxxxxx
#     message_body = data.get("Body")

#     latest_whatsapp_message = {
#         "from": from_number,
#         "message": message_body
#     }

#     print("Received WhatsApp message:", latest_whatsapp_message)

#     # Step 1: Send the message to Gemini via the ask API
#     try:
#         response = model.generate_content(message_body)
#         answer = response.text
#     except Exception as e:
#         answer = "Sorry, I couldn't process your message."

#     # Step 2: Send the answer back to the sender via Twilio
#     try:
#         sent = client.messages.create(
#             body=answer,
#             from_=TWILIO_WHATSAPP_NUMBER,
#             to=from_number
#         )
#         print(f"Sent reply to {from_number}: {answer}")
#         return {"status": "replied", "sid": sent.sid}
#     except Exception as e:
#         print(f"Failed to send reply: {e}")
#         return {"status": "error", "detail": str(e)}

@app.post("/receive-whatsapp")
async def receive_whatsapp(request: Request):
    global latest_whatsapp_message
    data = await request.form()
    from_number = data.get("From")
    message_body = data.get("Body")
    num_media = int(data.get("NumMedia", 0))

    latest_whatsapp_message = {
        "from": from_number,
        "message": message_body
    }

    print("Received WhatsApp message:", latest_whatsapp_message)

    try:
        if num_media > 0:
            media_url = data.get("MediaUrl0")
            media_type = data.get("MediaContentType0")

            # Download media from Twilio
            media_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            image_bytes = media_response.content

            # Send image to Gemini
            gemini_response = image_model.generate_content([
                {"text": message_body or "Describe this image"},
                {"inline_data": {"mime_type": media_type, "data": image_bytes}}
            ])
            answer = gemini_response.text
        else:
            # Text-only message
            response = model.generate_content(message_body)
            answer = response.text
    except Exception as e:
        print(f"Gemini processing failed: {e}")
        answer = "Sorry, I couldn't process your message or image."

    # Send response back via WhatsApp
    try:
        sent = client.messages.create(
            body=answer,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=from_number
        )
        print(f"Sent reply to {from_number}: {answer}")
        return {"status": "replied", "sid": sent.sid}
    except Exception as e:
        print(f"Failed to send reply: {e}")
        return {"status": "error", "detail": str(e)}



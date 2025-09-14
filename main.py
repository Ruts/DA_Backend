import random
from typing import List
from fastapi import FastAPI, File, Form, HTTPException, Request, Depends, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import google.generativeai as genai
from twilio.rest import Client
import requests
from io import BytesIO
from PIL import Image
from azure.cosmos import CosmosClient
from jose import jwt, JWTError
from datetime import date, datetime, timedelta
from uuid import uuid4
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

class SoilSample(BaseModel):
    id:str
    farmId: str
    date: date
    pH: float
    electricalConductivity: float
    organicMatter: float
    nitrogen: float
    phosphorus: float
    potassium: float
    calcium: float
    magnesium: float
    sulfur: float
    zinc: float
    iron: float
    soilTexture: str
    moistureContent: float

# Only load .env if running locally
if os.getenv("AZURE_ENVIRONMENT") != "production":
    load_dotenv()

ROLE_PROMPT = os.getenv("ROLE_PROMPT")
print(f"ROLE_PROMPT: {ROLE_PROMPT}")
IMAGE_ANALYSIS_PROMPTS = os.getenv("IMAGE_ANALYSIS_PROMPTS")
print(f"IMAGE_ANALYSIS_PROMPTS: {IMAGE_ANALYSIS_PROMPTS}")
SOIL_MONITORING_PROMPTS = os.getenv("SOIL_MONITORING_PROMPTS")
print(f"SOIL_MONITORING_PROMPTS: {SOIL_MONITORING_PROMPTS}")
RECOMMENDATION_PROMPS = os.getenv("RECOMMENDATION_PROMPS")
print(f"RECOMMENDATION_PROMPS: {RECOMMENDATION_PROMPS}")
SUMMARIZE = os.getenv("SUMMARIZE")

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-1.5-pro"
image_model = genai.GenerativeModel("gemini-2.5-flash-image-preview")  # for image input

# Twilio credentials (use environment variables in production)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

# At the top of your FastAPI file
latest_whatsapp_message = {"from": "", "message": ""}

TOKEN_SECRET_KEY = os.getenv("TOKEN_SECRET_KEY")
TOKEN_ALGORITHM = os.getenv("TOKEN_ALGORITHM")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
DA_FARM_IMAGES_KEY = os.getenv("DA_FARM_IMAGES_KEY")

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
DATABASE_NAME = os.getenv("DATABASE_NAME")
USERS_CONTAINER_NAME = "users"
FARMS_CONTAINER_NAME = "farms"
HARVESTS_CONTAINER_NAME = "harvests"
ENVIRONMENTS_CONTAINER_NAME = "environments"
GRID_CONTAINER_NAME = "GridImages"
SOIL_CONTAINER_NAME = "soil_monitors"
CHATS_CONTAINER_NAME = "chats"

cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.get_database_client(DATABASE_NAME)
users_container = database.get_container_client(USERS_CONTAINER_NAME)
farms_container = database.get_container_client(FARMS_CONTAINER_NAME)
harvests_container = database.get_container_client(HARVESTS_CONTAINER_NAME)
environments_container = database.get_container_client(ENVIRONMENTS_CONTAINER_NAME)
grid_container = database.get_container_client(GRID_CONTAINER_NAME)
soil_container = database.get_container_client(SOIL_CONTAINER_NAME)
chats_container = database.get_container_client(CHATS_CONTAINER_NAME)

blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client("grid-images")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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

@app.get("/latest-whatsapp")
async def get_latest_whatsapp():
    return latest_whatsapp_message

# @app.post("/receive-whatsapp")
# async def receive_whatsapp(request: Request):
#     global latest_whatsapp_message
#     data = await request.form()
#     from_number = data.get("From")
#     message_body = data.get("Body")
#     num_media = int(data.get("NumMedia", 0))

#     latest_whatsapp_message = {
#         "from": from_number,
#         "message": message_body
#     }

#     print("Received WhatsApp message:", latest_whatsapp_message)

#     try:
#         if num_media > 0:
#             media_url = data.get("MediaUrl0")
#             media_type = data.get("MediaContentType0")

#             # Download media from Twilio
#             media_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
#             image_bytes = media_response.content

#             # Send image to Gemini
#             gemini_response = image_model.generate_content([
#                 {"text": message_body or "Describe this image"},
#                 {"inline_data": {"mime_type": media_type, "data": image_bytes}}
#             ])
#             answer = gemini_response.text
#         else:
#             # Text-only message
#             response = model.generate_content(message_body)
#             answer = response.text
#     except Exception as e:
#         print(f"Gemini processing failed: {e}")
#         answer = "Sorry, I couldn't process your message or image."

#     # Send response back via WhatsApp
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

def send_verification_code(phone_number):
    code = str(random.randint(100000, 999999))
    client.messages.create(
        from_="whatsapp:+14155238886",
        to=f"whatsapp:{phone_number}",
        body=f"Your verification code is: {code}"
    )
    return code

@app.post("/create-account")
async def create_account(request: Request):
    data = await request.json()
    phone = data["phone"]
    name = data["name"]

    code = send_verification_code(phone)

    users_container.upsert_item({
        "id": phone,
        "phoneNumber": phone,
        "name": name,
        "verified": False,
        "code": code
    })

    return {"status": "code_sent"}
    
@app.post("/login")
async def login(request: Request):
    print(f"request: {request}")
    data = await request.json()
    print(f"data: {data}")
    phone = data["phone"]

    user = users_container.read_item(item=phone, partition_key=phone)
    print(f"user: {user}")

    if user["verified"]:
        code = send_verification_code(phone)
        print(f"code: {code}")
        user["code"] = code
        users_container.upsert_item(user)
        return {"status": "verify", "user": user}
    else:
        return {"status": "not_verified"}

def verify_token(token: str):
    try:
        payload = jwt.decode(token, TOKEN_SECRET_KEY, algorithms=[TOKEN_ALGORITHM])
        return payload
    except JWTError:
        return None
    
def create_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode(to_encode, TOKEN_SECRET_KEY, algorithm=TOKEN_ALGORITHM)

@app.post("/verify-code")
async def verify_code(request: Request):
    data = await request.json()
    phone = data["phone"]
    code = data["code"]

    user = users_container.read_item(item=phone, partition_key=phone)

    if user["code"] == code:
        user["verified"] = True
        users_container.upsert_item(user)
        token = create_token({"phone": phone, "name": user["name"]})
        return {"status": "verified", "token": token}
    else:
        raise HTTPException(status_code=401, detail="Invalid code")

async def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    return verify_token(token)

@app.get("/dashboard-data")
async def dashboard_data(user=Depends(get_current_user)):
    # `user` contains decoded JWT payload
    return {"message": f"Welcome {user['name']}!", "phone": user["phone"]}

@app.post("/farms")
async def create_farm(request: Request, user=Depends(get_current_user)):
    data = await request.json()
    farm = {
        "id": str(uuid4()),
        "owner": user["phone"],
        "name": data["name"],
        "crop": data["crop"],
        "acreage": data["acreage"],
        "coordinates": data["coordinates"]
    }
    farms_container.upsert_item(farm)
    return {"status": "created", "farm": farm}

@app.get("/farms")
async def get_user_farms(user=Depends(get_current_user)):
    query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    farms = list(farms_container.query_items(query, enable_cross_partition_query=True))
    return farms

@app.post("/harvests")
async def add_harvest(request: Request, user=Depends(get_current_user)):
    data = await request.json()
    harvest = {
        "id": str(uuid4()),
        "farmId": data["farmId"],
        "owner": user["phone"],
        "date": data["date"],
        "yield": data["yield"],
        "crop": data["crop"]
    }
    harvests_container.upsert_item(harvest)
    return {"status": "saved", "harvest": harvest}

@app.get("/harvests")
async def get_user_harvests(user=Depends(get_current_user)):
    query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    return list(harvests_container.query_items(query, enable_cross_partition_query=True))

@app.post("/environments")
async def add_environmental_data(request: Request, user=Depends(get_current_user)):
    data = await request.json()
    record = {
        "id": str(uuid4()),
        "farmId": data["farmId"],
        "owner": user["phone"],
        "date": data["date"],
        "soilMoisture": data["soilMoisture"],
        "soilPH": data.get("soilPH"),
        "temperature": data.get("temperature")
    }
    environments_container.upsert_item(record)
    return {"status": "saved", "data": record}

@app.get("/environments")
async def get_environmental_data(user=Depends(get_current_user)):
    query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    return list(environments_container.query_items(query, enable_cross_partition_query=True))

@app.post("/grid-images")
async def upload_multiple_grid_images(
    farmId: str = Form(...),
    dateUploaded: date = Form(...),
    files: List[UploadFile] = File(...),
    user=Depends(get_current_user)
):
    image_urls = []

    for file in files:
        blob_name = f"{farmId}_{dateUploaded}_{uuid4()}.jpg"
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file.file, overwrite=True)
        image_urls.append(blob_client.url)

    metadata = {
        "id": str(uuid4()),
        "farmId": farmId,
        "owner": user["phone"],
        "date": str(dateUploaded),
        "imageUrl": image_urls  # ✅ Now an array of URLs
    }

    grid_container.upsert_item(metadata)

    await analyze_farm_day(farmId, dateUploaded)
    return {"status": "uploaded", "imageUrl": image_urls}

@app.get("/grid-images")
async def get_grid_images(user=Depends(get_current_user)):
    query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    items = list(grid_container.query_items(query, enable_cross_partition_query=True))

    for item in items:
        if isinstance(item["imageUrl"], list):
            item["imageUrl"] = [
                generate_sas_url(blob_url.split("/")[-1].split("?")[0])
                for blob_url in item["imageUrl"]
            ]
        else:
            item["imageUrl"] = generate_sas_url(item["imageUrl"].split("/")[-1].split("?")[0])

    return items

def generate_sas_url(blob_name: str) -> str:
    sas_token = generate_blob_sas(
        account_name="dafarmimages",
        container_name="grid-images",
        blob_name=blob_name,
        account_key=DA_FARM_IMAGES_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    url_base = os.getenv("IMAGE_STORAGE_URL")
    return f"{url_base}/{blob_name}?{sas_token}"

@app.post("/soil-samples")
async def store_soil_sample(sample: SoilSample, user=Depends(get_current_user)):
    print(f"sample: {sample}")
    sample_doc = sample.dict()
    sample_doc["owner"] = user["phone"]
    sample_doc["id"] = str(uuid4())
    sample_doc = serialize_dates(sample_doc)
    print(f"sample_doc: {sample_doc}")
    soil_container.upsert_item(sample_doc)

    await analyze_farm_day(sample["farmId"], sample["date"])
    return {"status": "stored"}

@app.get("/soil-samples")
async def get_soil_samples(
    farmId: Optional[str] = Query(None),
    startDate: Optional[date] = Query(None),
    endDate: Optional[date] = Query(None),
    user=Depends(get_current_user)
):
    base_query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    
    if farmId:
        base_query += f" AND c.farmId = '{farmId}'"
    if startDate:
        base_query += f" AND c.date >= '{startDate}'"
    if endDate:
        base_query += f" AND c.date <= '{endDate}'"

    items = list(soil_container.query_items(base_query, enable_cross_partition_query=True))
    return items

def serialize_dates(obj):
    for key, value in obj.items():
        if isinstance(value, (date, datetime)):
            obj[key] = value.isoformat()
    return obj

@app.post("/chat")
async def handle_chat(request: Request):
    data = await request.json()
    user = data.get("user")  # e.g. "+254712345678"
    message = data.get("message")

    # Step 1: Store incoming message
    user_msg = {
        "id": str(uuid4()),
        "user": user,
        "role": "user",
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(user_msg)

    # Step 2: Fetch full history
    query = f"SELECT * FROM c WHERE c.user = '{user}' ORDER BY c.timestamp ASC"
    history = list(chats_container.query_items(query, enable_cross_partition_query=True))

    # Step 3: Format for LLM
    formatted = [{"role": h["role"], "content": h["message"]} for h in history]

    # Step 4: Send to LLM
    response = model.generate_content(formatted)
    answer = response.text

    # Step 5: Store assistant reply
    assistant_msg = {
        "id": str(uuid4()),
        "user": user,
        "role": "assistant",
        "message": answer,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(assistant_msg)

    return {"answer": answer}

@app.post("/receive-whatsapp")
async def receive_whatsapp(request: Request):
    data = await request.form()
    from_number = data.get("From").replace("whatsapp:", "")
    message_body = data.get("Body")

    # Forward to chat handler
    chat_response = await handle_chat(Request(scope={"type": "http"}, receive=lambda: None))
    answer = chat_response["answer"]

    # Send reply via Twilio
    try:
        sent = client.messages.create(
            body=answer,
            from_=TWILIO_WHATSAPP_NUMBER,
            to="whatsapp:" + from_number
        )
        return {"status": "replied", "sid": sent.sid}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
@app.get("/chat-history")
async def get_chat_history(user: str):
    query = f"SELECT * FROM c WHERE c.user = '{user}' ORDER BY c.timestamp ASC"
    history = list(chats_container.query_items(query, enable_cross_partition_query=True))
    return history

@app.post("/analyze-farm-day")
async def analyze_farm_day(
    farmId: str = Form(...),
    date: date = Form(...),
    user=Depends(get_current_user)
):
    # Step 1: Fetch grid images
    img_query = f"""
        SELECT * FROM c WHERE c.owner = '{user['phone']}' AND c.farmId = '{farmId}' AND c.date = '{date}'
    """
    images = list(grid_container.query_items(img_query, enable_cross_partition_query=True))

    # Step 2: Fetch soil samples
    soil_query = f"""
        SELECT * FROM c WHERE c.owner = '{user['phone']}' AND c.farmId = '{farmId}' AND c.date = '{date}'
    """
    soil = list(soil_container.query_items(soil_query, enable_cross_partition_query=True))

    if not images or not soil:
        return {"status": "incomplete", "message": "Missing either grid images or soil data for this date."}

    # Step 3: Generate SAS URLs
    image_urls = []
    for img in images:
        for url in img["imageUrl"]:
            blob_name = url.split("/")[-1].split("?")[0]
            image_urls.append(generate_sas_url(blob_name))

    # Step 4: Build prompt
    role_prompt = os.getenv("ROLE_PROMPT")
    image_prompt = os.getenv("IMAGE_ANALYSIS_PROMPTS")
    soil_prompt = os.getenv("SOIL_MONITORING_PROMPTS")
    recommendation_prompt = os.getenv("RECOMMENDATION_PROMPS")
    summarize_prompt = os.getenv("SUMMARIZE")

    soil_data = soil[0]  # Assuming one sample per day
    soil_text = "\n".join([
        f"pH: {soil_data['pH']}",
        f"EC: {soil_data['electricalConductivity']} dS/m",
        f"Organic Matter: {soil_data['organicMatter']}%",
        f"N: {soil_data['nitrogen']} ppm",
        f"P: {soil_data['phosphorus']} ppm",
        f"K: {soil_data['potassium']} ppm",
        f"Ca: {soil_data['calcium']} ppm",
        f"Mg: {soil_data['magnesium']} ppm",
        f"S: {soil_data['sulfur']} ppm",
        f"Zn: {soil_data['zinc']} ppm",
        f"Fe: {soil_data['iron']} ppm",
        f"Texture: {soil_data['soilTexture']}",
        f"Moisture: {soil_data['moistureContent']}%"
    ])

    full_prompt = f"""
{role_prompt}

{image_prompt}

Images:
{chr(10).join(image_urls)}

{soil_prompt}

Soil Data:
{soil_text}

{recommendation_prompt}

{summarize_prompt}
    """

    # Step 5: Send to Google LLM
    response = model.generate_content(full_prompt)
    answer = response.text

    # Step 6: Store response
    analysis_doc = {
        "id": str(uuid4()),
        "user": user,
        "role": "assistant",
        "message": answer,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(analysis_doc)

    return {"status": "complete", "response": answer}


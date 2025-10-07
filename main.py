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
import CropYieldModel as model_utils
import joblib

app = FastAPI(title="Crop Yield Predictor")

# Load preprocessing and models once
ct, crops, soil_dim = model_utils.load_preprocessing("data/soil_data.csv")
models_by_crop = model_utils.load_models(crops, soil_dim)
MODEL_DIR = "saved_models"

# Class for storing data used in yields prediction
class SoilInput(BaseModel):
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
    crop_type: str
    image_paths: List[str]

# Class for storing data used when saving the data to the DB
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
IMAGE_ANALYSIS_PROMPTS = os.getenv("IMAGE_ANALYSIS_PROMPTS")
SOIL_MONITORING_PROMPTS = os.getenv("SOIL_MONITORING_PROMPTS")
RECOMMENDATION_PROMPS = os.getenv("RECOMMENDATION_PROMPS")
SUMMARIZE = os.getenv("SUMMARIZE")

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
image_model = genai.GenerativeModel("gemini-2.5-flash-image-preview") 

# Twilio credentials
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = FastAPI()

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
grid_images_client = blob_service_client.get_container_client("grid-images")
user_images_client = blob_service_client.get_container_client("user-images")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API to send question to LLM
@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")

    response = model.generate_content(question)
    answer = response.text

    return {"answer": answer}

# API to send messages to WhatsApp API
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

# API to fecth the latest whatsapp messages
@app.get("/latest-whatsapp")
async def get_latest_whatsapp():
    return latest_whatsapp_message

def send_verification_code(phone_number):
    code = str(random.randint(100000, 999999))
    client.messages.create(
        from_="whatsapp:+14155238886",
        to=f"whatsapp:{phone_number}",
        body=f"Your verification code is: {code}"
    )
    return code

# API to create anew account
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
    
# API to log in to app
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

# API to verify thecode sent
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

# API to get the name and number of the user
@app.get("/dashboard-data")
async def dashboard_data(user=Depends(get_current_user)):
    # `user` contains decoded JWT payload
    return {"message": f"Welcome {user['name']}!", "phone": user["phone"]}

# API to add the farm data
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

# API to get the farm data
@app.get("/farms")
async def get_user_farms(user=Depends(get_current_user)):
    query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    farms = list(farms_container.query_items(query, enable_cross_partition_query=True))
    return farms

# API to add a harvest
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

# API to fetch harvest data
@app.get("/harvests")
async def get_user_harvests(user=Depends(get_current_user)):
    query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    return list(harvests_container.query_items(query, enable_cross_partition_query=True))

# API to add environmental conditions
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

# API to fetch th environment data
@app.get("/environments")
async def get_environmental_data(user=Depends(get_current_user)):
    query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}'"
    return list(environments_container.query_items(query, enable_cross_partition_query=True))

# API to add farm grid images
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
        blob_client = grid_images_client.get_blob_client(blob_name)
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

# API to fetch farm grid images
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

# API to generate a url to fetch images for farm images
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

# API to generate a url to fetch user submitted images
def generate_sas_url_user_image(blob_name: str) -> str:
    sas_token = generate_blob_sas(
        account_name="dafarmimages",
        container_name="user-images",
        blob_name=blob_name,
        account_key=DA_FARM_IMAGES_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    url_base = os.getenv("USER_IMAGE_STORAGE_URL")
    return f"{url_base}/{blob_name}?{sas_token}"

# API to add soil monitoring data
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

# API to get soil data
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

# Method to handle chat data before upload
async def handle_chat_payload(data: dict):
    user = data.get("owner")  # e.g. "+254712345678"
    message = data.get("message")
    print(f"data: {data}")
    print(f"user: {user}")
    print(f"message: {message}")

    # Step 2: Fetch full history
    phone = user["phone"]
    query = f"SELECT * FROM c WHERE c.user = '{phone}' ORDER BY c.timestamp ASC"
    history = list(chats_container.query_items(query, enable_cross_partition_query=True))

    # Step 3: Format for LLM
    # formatted = [{"role": h["role"], "content": h["message"]} for h in history]
    formatted = [
        {"role": h["role"], "parts": [{"text": h["message"]}]}
        for h in history
    ]

    formatted = ""
    for h in history:
        formatted = formatted + " role: " + h["role"] + " text: " + h["message"] + " timestamp: " + h["timestamp"] + "\n"

    print(f"formatted: {formatted}")

    # Step 1: Store incoming message
    user_msg = {
        "id": str(uuid4()),
        "user": user["phone"],
        "role": "user",
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(user_msg)

    return {"formatted": formatted}

# API to add chat data
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    return await handle_chat_payload(data)

# API to manage received whatsapp messages
@app.post("/receive-whatsapp")
async def receive_whatsapp(request: Request):
    form = await request.form()
    from_number = form.get("From", "").replace("whatsapp:+", "")
    num_media = int(form.get("NumMedia", 0))
    message_body = form.get("Body", "")
    print(f"data: {form}")
    print(f"from_number: {from_number}")
    print(f"num_media: {num_media}")
    print(f"message_body: {message_body}")

    # Forward to chat handler
    chat_data = {
        "owner": {"phone": from_number},
        "message": message_body
    }
    chat_formatted = await handle_chat_payload(chat_data)
    formatted = "\n".join([
        f"Answer the users Current message. Chat history has been included for context only",
        f"Chat History: {chat_formatted['formatted']}",
        f"Current message: {message_body}"
    ])
    answer = "LLM call failed"
    print(f"answer: {answer}")
    print(f"formatted: {formatted}")

    image_urls = []
    media_descriptions = []

    for i in range(num_media):
        media_url = form.get(f"MediaUrl{i}")
        media_type = form.get(f"MediaContentType{i}")

        try:
            media_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            image_bytes = media_response.content

            blob_name = f"{from_number}_{datetime.utcnow().isoformat()}_{uuid4()}.jpg"
            blob_client = user_images_client.get_blob_client(blob_name)
            blob_client.upload_blob(image_bytes, overwrite=True)

            sas_url = generate_sas_url_user_image(blob_name)
            image_urls.append(sas_url)
            media_descriptions.append(f"Image {i+1}: {sas_url} (type: {media_type})")

        except Exception as e:
            print(f"Failed to process media {i}: {e}")

    # Build message for Gemini
    full_message = f"{formatted}\n\nAttached images:\n" + "\n".join(media_descriptions)
    print(f"full_message: {full_message}")

    try:
        if num_media > 0:

            media_url = form.get("MediaUrl0")
            media_type = form.get("MediaContentType0")

            media_response = requests.get(media_url, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
            image_bytes = media_response.content

            blob_name = f"{from_number}_{datetime.utcnow().isoformat()}_{uuid4()}.jpg"
            blob_client = user_images_client.get_blob_client(blob_name)
            blob_client.upload_blob(image_bytes, overwrite=True)

            gemini_response = image_model.generate_content([
                {
                    "role": "user",
                    "parts": [
                        {"text": full_message or "Describe this image"},
                        {"inline_data": {
                            "mime_type": media_type,
                            "data": image_bytes
                        }}
                    ]
                }
            ])
            answer = gemini_response.text
        else:
            response = model.generate_content(formatted)
            answer = response.text

    except Exception as e:
        print(f"Gemini processing failed: {e}")
        answer = "Sorry, I couldn't process your message or image."

    assistant_msg = {
        "id": str(uuid4()),
        "user": from_number,
        "role": "assistant",
        "message": answer,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(assistant_msg)
    print(f"final answer: {answer}")

    try:
        sent = client.messages.create(
            body=answer,
            from_=TWILIO_WHATSAPP_NUMBER,
            to="whatsapp:+" + from_number
        )
        return {"status": "replied", "sid": sent.sid}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
# API to fech chat history
@app.get("/chat-history")
async def get_chat_history(user=Depends(get_current_user)):
    print(f"user: {user}")
    phone = user["phone"]
    query = f"SELECT * FROM c WHERE c.user = '{phone}' ORDER BY c.timestamp ASC"
    print(f"query: {query}")
    history = list(chats_container.query_items(query, enable_cross_partition_query=True))
    print(f"history: {history}")
    return history

# API to analyse farm data
@app.post("/analyze-farm-day")
async def analyze_farm_day(
    farmId: str = Form(...),
    date: date = Form(...),
    user=Depends(get_current_user)
):
    print(f"farmId: {farmId}")
    print(f"date: {date}")

    farm_query = f"SELECT * FROM c WHERE c.owner = '{user['phone']}' AND c.id = '{farmId}'"
    farms = list(farms_container.query_items(farm_query, enable_cross_partition_query=True))
    
    img_query = f"""
        SELECT * FROM c WHERE c.owner = '{user['phone']}' AND c.farmId = '{farmId}' AND c.date = '{date}'
    """
    images = list(grid_container.query_items(img_query, enable_cross_partition_query=True))

    soil_query = f"""
        SELECT * FROM c WHERE c.owner = '{user['phone']}' AND c.farmId = '{farmId}' AND c.date = '{date}'
    """
    soil = list(soil_container.query_items(soil_query, enable_cross_partition_query=True))

    if not images or not soil:
        print(f"not images or not soil")
        return {"status": "incomplete", "message": "Missing either grid images or soil data for this date."}

    image_urls = []
    for img in images:
        for url in img["imageUrl"]:
            blob_name = url.split("/")[-1].split("?")[0]
            image_urls.append(generate_sas_url(blob_name))

    role_prompt = os.getenv("ROLE_PROMPT")
    image_prompt = os.getenv("IMAGE_ANALYSIS_PROMPTS")
    soil_prompt = os.getenv("SOIL_MONITORING_PROMPTS")
    recommendation_prompt = os.getenv("RECOMMENDATION_PROMPS")
    summarize_prompt = os.getenv("SUMMARIZE")
    predict_prompt = os.getenv("PREDICT")

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

    soilInput = SoilInput(
        pH=soil_data['pH'],
        electricalConductivity=soil_data['electricalConductivity'],
        organicMatter=soil_data['organicMatter'],
        nitrogen=soil_data['nitrogen'],
        phosphorus=soil_data['phosphorus'],
        potassium=soil_data['potassium'],
        calcium=soil_data['calcium'],
        magnesium=soil_data['magnesium'],
        sulfur=soil_data['sulfur'],
        zinc=soil_data['zinc'],
        iron=soil_data['iron'],
        soilTexture=soil_data['soilTexture'],
        moistureContent=soil_data['moistureContent'],
        crop_type=farms[0]["crop"],
        image_paths=image_urls
    )

    predicted_yield = predict_yield_method(soilInput)
    print(f"predicted_yield: {predicted_yield}")

    full_prompt = f"""
{role_prompt}

{image_prompt}

Images:
{chr(10).join(image_urls)}

{soil_prompt}

Soil Data:
{soil_text}

{recommendation_prompt}

{predict_prompt}: {predicted_yield}
    """

    user_msg = {
        "id": str(uuid4()),
        "user": user["phone"],
        "role": "user",
        "message": full_prompt,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(user_msg)

    try:
        print(f"full_prompt: {full_prompt}")
        response = model.generate_content(full_prompt)
        answer = response.text
        print(f"answer: {answer}")
    except Exception as e:
        print(f"Content generation failed: {e}")
        raise HTTPException(status_code=500, detail="AI generation failed")

    analysis_doc = {
        "id": str(uuid4()),
        "user": user['phone'],
        "role": "assistant",
        "message": answer,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(analysis_doc)

    responseSummary = model.generate_content(summarize_prompt + ": " + answer)
    print(f"summarize_prompt: {summarize_prompt}")
    answerSummary = responseSummary.text
    print(f"answerSummary: {answerSummary}")

    analysis_doc = {
        "id": str(uuid4()),
        "user": user['phone'],
        "role": "assistant",
        "message": "Answer Summary: " + answerSummary,
        "timestamp": datetime.utcnow().isoformat()
    }
    chats_container.upsert_item(analysis_doc)

    try:
        sent = client.messages.create(
            body=answerSummary,
            from_=TWILIO_WHATSAPP_NUMBER,
            to="whatsapp:+" + user['phone']
        )
        print(f"message sent; ", sent)
    except Exception as e:
        sent = client.messages.create(
            body="Error sending message. try again.",
            from_=TWILIO_WHATSAPP_NUMBER,
            to="whatsapp:+" + user['phone']
        )
        print(f"message sent; ", sent)
        print(f"status: ", "error sending message ", "detail: ", str(e))

    return {"status": "complete", "response": answer}

# API to predict crop yield based on soil monitoring and image data
@app.post("/predict")
def predict_yield(input: SoilInput):
    return predict_yield_method(input)

def predict_yield_method(input: SoilInput) -> str:
    print(f"input; ", input)
    model = models_by_crop.get(input.crop_type)
    if not model:
        return {"error": f"No model found for crop type: {input.crop_type}"}
    soil_dict = input.dict(exclude={"crop_type", "image_paths"})
    target_scalers = {}
    scaler = joblib.load(f"{MODEL_DIR}/{input.crop_type}_scaler.pkl")
    target_scalers[input.crop_type] = scaler
    yield_kg = model_utils.predict_yield(soil_dict, input.image_paths, input.crop_type, models_by_crop, ct, target_scalers)
    return {"predicted_yield_kg": round(yield_kg, 2)}
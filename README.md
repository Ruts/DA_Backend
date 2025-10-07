# 🌱 Digital Agronomist API

Available at: https://dabackend-c8fcgxdhdyfff7am.canadacentral-01.azurewebsites.net/

This project is a FastAPI-based backend designed for digital agriculture platforms.
It integrates soil monitoring, crop advisory, farm management, image analysis, and WhatsApp-based farmer communication using Google Gemini LLMs, Twilio, and Azure services.

## 🚀 Features

AI-Powered Advisory

Uses Google Gemini (gemini-1.5-flash, gemini-2.5-flash-image-preview) to analyze farm images, soil samples, and provide tailored recommendations.

Farmer Communication

WhatsApp integration via Twilio for sending/receiving messages.

AI-powered chatbot for interactive farmer support.

Farm & Harvest Management

Create and manage farms.

Record harvest yields.

Track environmental data (soil moisture, pH, temperature).

Soil Monitoring

Store and query detailed soil test results.

AI-assisted interpretation of soil data.

Satellite / Drone Image Analysis

Upload and store field grid images.

Automatic SAS URL generation with Azure Blob Storage.

AI analysis combining soil + image data.

Secure Authentication

JWT-based user authentication and authorization.

WhatsApp verification for farmer accounts.

Data Storage

Azure Cosmos DB for structured farm, user, harvest, and soil monitoring data.

## 📦 Tech Stack

Framework: FastAPI

AI Model: Google Gemini

Messaging: Twilio WhatsApp API

Database: Azure Cosmos DB

Storage: Azure Blob Storage

Auth: JWT (python-jose)

Image Processing: Pillow (PIL)

## ⚙️ Setup & Installation
1. Clone Repository
git clone https://github.com/your-username/farm-management-api.git
cd farm-management-api

2. Create & Activate Virtual Environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3. Install Dependencies
pip install -r requirements.txt

4. Environment Variables

Create a .env file in the project root with the following:

### Google Gemini
GOOGLE_API_KEY=your_google_api_key
ROLE_PROMPT="You are an agronomist assistant..."
IMAGE_ANALYSIS_PROMPTS="Analyze crop images..."
SOIL_MONITORING_PROMPTS="Analyze soil data..."
RECOMMENDATION_PROMPS="Provide farm recommendations..."
SUMMARIZE="Summarize insights..."

### Twilio WhatsApp
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

### Azure
AZURE_STORAGE_CONNECTION_STRING=your_storage_conn_string
DA_FARM_IMAGES_KEY=your_blob_account_key
IMAGE_STORAGE_URL=https://yourstorageaccount.blob.core.windows.net/grid-images
COSMOS_ENDPOINT=your_cosmos_endpoint
COSMOS_KEY=your_cosmos_key
DATABASE_NAME=farm_db

### JWT
TOKEN_SECRET_KEY=your_secret_key
TOKEN_ALGORITHM=HS256

5. Run Server
uvicorn main:app --reload


API available at:
👉 http://127.0.0.1:8000

## 📖 API Endpoints
### Authentication

POST /create-account → Register farmer (sends WhatsApp verification code).

POST /login → Farmer login.

POST /verify-code → Verify code and issue JWT.

GET /dashboard-data → Get user dashboard (JWT required).

### Farms & Harvests

POST /farms → Create a farm.

GET /farms → Get user farms.

POST /harvests → Add harvest record.

GET /harvests → List harvests.

### Soil & Environment

POST /soil-samples → Store soil sample.

GET /soil-samples → Query soil data by farm/date.

POST /environments → Add environmental data.

GET /environments → Get environmental records.

### Images

POST /grid-images → Upload multiple grid images.

GET /grid-images → Retrieve farm images (SAS URL).

### AI Chat & Analysis

POST /ask → Ask AI a question.

POST /chat → Farmer–AI conversation.

GET /chat-history?user={phone} → Retrieve chat history.

POST /analyze-farm-day → Run AI analysis for a given farm & date.

### WhatsApp

POST /send-whatsapp → Send message via WhatsApp.

POST /receive-whatsapp → Receive and auto-reply to WhatsApp messages.

GET /latest-whatsapp → Get latest WhatsApp message.

### Crop Yield Prediction
POST /predict → Predict crop yield from soil data and aerial images using prebuild models

## 🔐 Security Notes

Always store secrets in .env, never in code.

Use HTTPS in production.

Rotate Twilio & Azure keys regularly.

## 📌 Future Improvements

Role-based access control (farmers vs. agronomists).

Offline-first support for rural deployments.

Integration with satellite NDVI imagery.

Advanced analytics dashboards.

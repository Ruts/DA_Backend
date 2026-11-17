# 🌱 Digital Agronomist API

Available at: https://dabackend-c8fcgxdhdyfff7am.canadacentral-01.azurewebsites.net/

This project is a FastAPI-based backend designed for digital agriculture platforms.
It integrates soil monitoring, crop advisory, farm management, image analysis, and WhatsApp-based farmer communication using Google Gemini, Meta's direct WhatsApp Cloud API, and Azure services.

## WhatsApp-first service

Farmers start by messaging the service number. Guided registration captures consent,
English/Kiswahili, name, farm name, location pin (optional until weather is enabled),
crop, acres and planting date. Returning farmers can ask questions or send photos
with captions. Advice uses their selected farm and its recent chat history, with
uncertainty and referral guidance rather than a confirmed diagnosis.

Commands: `HELP`, `LOGIN`, `FARMS`, `FARM 1`, `ADD FARM`, `WEATHER`,
`WEATHER NOW`, `WEATHER DAILY 07:00`, `WEATHER ALTERNATE 07:00`,
`WEATHER WEEKLY MON 07:00`, and `STOP`.

The portal's **Profile & Weather** page edits the same profile and farm records.
Staff can look up a farmer by phone only when their own verified phone is explicitly
listed in server-side `STAFF_PHONES`; the portal cannot grant staff access.
Profile names displayed in an existing JWT/sidebar refresh on the next login.

### Staff conversation inbox

The portal's **Staff Inbox** is visible only to phones in server-side `STAFF_PHONES`.
It shows all captured WhatsApp inbound/outbound interactions (including onboarding,
photo captions and weather), with phone filtering, pagination and authenticated photo
viewing. Login codes are redacted; `accepted` means accepted by Meta, not delivered.
The **Advice / portal history** source exposes existing stored advice conversations.
New WhatsApp auditing starts at this deployment; it cannot reconstruct old messages
that were never stored. Meta can expire media, so old photo previews may be unavailable.

Create **`whatsapp_interactions`**, partition **`/owner`**, default indexing, no TTL.
Unlike the short-lived job queue, this is the staff interaction archive. Apply the
project's retention/deletion policy to it before the pilot. Do not grant Cosmos access
to ordinary users. The staff endpoints enforce the allowlist on every request.

For WhatsApp phone-app viewing, Meta supports **Business App coexistence** through its
[Embedded Signup flow](https://developers.facebook.com/documentation/business-messaging/whatsapp/embedded-signup/onboarding-business-app-users).
This is not ordinary linked-device access for a Cloud-API-only number. Meta requires
a Solution Partner/Tech Provider integration and additional history/echo webhooks.
Those coexistence webhooks and human-agent handoff are not implemented here.
If phone-app viewing is essential, arrange coexistence eligibility/onboarding **before**
registering the new number as API-only. Do not independently register that API-only
number in the Business app. The direct Meta API plus staff portal is the supported path.

### Register the new number directly with Meta

1. Follow [Meta's Cloud API setup](https://developers.facebook.com/docs/whatsapp/cloud-api/get-started):
   create a Meta developer app with the WhatsApp use case and connect a business
   portfolio and WhatsApp Business Account. Complete Meta's applicable verification,
   publishing and payment requirements. Meta messaging fees still apply; Twilio is not used.
2. In WhatsApp Manager, add the new number and verify ownership by SMS or voice.
   The SIM must receive the verification code. Do not register it in the ordinary
   WhatsApp app first. Copy its **phone-number ID** and **WABA ID**, not just the
   human-readable phone number.
3. Generate a system-user access token, assign the app and WhatsApp account assets,
   and grant `whatsapp_business_messaging` and `whatsapp_business_management`
   (plus business-management permissions if required by Meta's setup).
   Store tokens and the app secret in deployment secrets, never Git or chat.
4. Configure the settings below. Use an API version supported by your app.
   Run `python meta_setup.py check` to verify the configured phone-number ID
   actually belongs to the intended service number.
5. To register the verified number, temporarily set `META_REGISTRATION_PIN` to a
   six-digit two-step-verification PIN, then run `python meta_setup.py register`.
   Keep the PIN in a password manager and remove it from the shell environment.
   Registration is a real external write; run it deliberately, not on server startup.
   See [Meta's registration requirements](https://developers.facebook.com/documentation/business-messaging/whatsapp/business-phone-numbers/registration).
6. Deploy the backend and configure Meta's callback as
   `https://<backend-host>/whatsapp/webhook`, with the configured verify token.
   Subscribe to the `messages` webhook field, then run `python meta_setup.py subscribe`
   to subscribe the app to the WABA.
7. Submit the weather template in both **English (`en`) and Swahili (`sw`)**.
   Example body: `Your requested farm weather forecast: {{1}}`.
   Include a realistic forecast as the sample variable. Meta determines template
   categorisation and approval; use the approved names in the settings below.
   Scheduled forecasts always use templates, including outside the 24-hour service window.
   Optionally approve an authentication template with a copy-code button for portal
   login; otherwise farmers send `LOGIN` in WhatsApp and enter that code in the portal.
8. Before inviting farmers, test with a controlled phone: registration, portal login,
   profile/farm editing, a text question, a captioned photo, forecast opt-in,
   a scheduled template, a schedule change and `STOP`.

### Durable jobs and forecast delivery

Create a Cosmos container named **`whatsapp_jobs`** in the existing database:
partition key **`/owner`**, default indexing enabled, and default TTL **`-1`**
(individual completed jobs expire after eight days to cover webhook redelivery). Restrict access to service operators.
Keep existing containers and their partition keys unchanged.

Run the API with `uvicorn main:app --host 0.0.0.0 --port 8000`.
Run **exactly one** separate, continuously running worker with
`python whatsapp_worker.py`, using the same environment and database. A restart-enabled,
always-on worker host is required; a local terminal or an App Service that sleeps is
not sufficient. Do not scale the worker beyond one replica: per-farmer conversation
ordering currently relies on a single consumer. Monitor worker restarts and queue age.
The existing GitHub workflow deploys only the API; provision the worker separately
before enabling the webhook for real users.

An alternative worker host is a separate Linux App Service on the existing Basic plan,
with Always On enabled, a single instance, and startup command:
`python -m uvicorn worker_app:app --host 0.0.0.0 --port 8000 --workers 1`.
Use `/health` to verify its last successful polling tick. Never run this startup command
on the API app or enable a second standalone worker simultaneously.

The webhook validates Meta's HMAC signature and phone-number ID, stores each message
durably, and acknowledges it without waiting for AI. The worker polls every minute.
Forecasts are off by default; enabling requires explicit consent and farm coordinates.
Frequency, local time, IANA timezone and weekly weekday are editable in the portal.
Every-two-day schedules advance by calendar days, not worker uptime. Missed forecasts
are coalesced rather than sending a backlog. Schedule changes invalidate queued jobs.

Monitor jobs with `pending`, `failed`, `sending` or `delivery_unknown` status and
Meta delivery-failure logs. Pre-send failures retry up to five times with backoff.
An earlier retrying or failed inbound job blocks that farmer's later messages to
preserve registration order; investigate failed jobs promptly. Other farmers continue.
An uncertain send is deliberately **not** retried automatically: check Meta's
message/delivery records before manually reconciling it to avoid duplicate messages.
If the worker crashes during `sending`, inspect that job the same way. This is not
an exactly-once delivery guarantee. Delivery may lag the selected minute while the
single worker handles other messages; size the pilot and monitor latency accordingly.

Forecasts use Open-Meteo's next three days of temperature and rainfall data.
Only coordinates are sent to the weather provider; farm photos go to the configured
AI provider for analysis. Open-Meteo's public endpoint is for eligible non-commercial
use; review its current terms and provision a commercial plan/endpoint before charging
for this service. Photos are processed transiently; chat records retain the Meta media
ID and caption, not an independently archived image. Establish a retention/deletion
policy for farmer and chat records before a real pilot.

Tests: `python -m pip install -r requirements-dev.txt`, then
`python -m unittest -v test_farmer_service`. These tests use fake storage and mocked
providers; they do not register a number or send real messages.

The legacy yield model loads only when `ENABLE_EXPERIMENTAL_YIELD=true`; leave it
disabled until its missing data, optional ML dependencies and field validation are
addressed. Ordinary WhatsApp photo advice does not depend on yield prediction.

## 🚀 Features

AI-Powered Advisory

Uses the configurable `GEMINI_MODEL` for multimodal crop advice.

Farmer Communication

Direct Meta WhatsApp Cloud API for receiving and sending messages.

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

Messaging: Meta WhatsApp Cloud API

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

### Meta WhatsApp and portal configuration
```text
WHATSAPP_BUSINESS_NUMBER=<international-digits-without-plus>
META_PHONE_NUMBER_ID=<Meta-phone-number-ID>
META_WABA_ID=<WhatsApp-Business-Account-ID>
META_GRAPH_VERSION=<supported-version-for-your-app>
META_ACCESS_TOKEN=<system-user-token>
META_APP_SECRET=<Meta-app-secret>
META_VERIFY_TOKEN=<random-private-webhook-verification-token>
META_WEATHER_TEMPLATE_EN=<approved-English-weather-template>
META_WEATHER_TEMPLATE_SW=<approved-Swahili-weather-template>
META_LOGIN_TEMPLATE=<optional-approved-authentication-template>
META_LOGIN_LANGUAGE=en_US
STAFF_PHONES=<comma-separated-international-staff-phone-numbers>
PORTAL_ORIGINS=https://<portal-host>,http://localhost:5173
GEMINI_MODEL=<supported-multimodal-model-ID>
```

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

POST /create-account → Returns instructions to register on WhatsApp instead.

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

POST /chat → Store authenticated farmer chat.

GET /chat-history → Retrieve the authenticated farmer's chat history.

POST /analyze-farm-day → Run AI analysis for a given farm & date.

### WhatsApp

GET /service-info → Public service number and WhatsApp entry link.

GET /whatsapp/webhook → Meta webhook verification.

POST /whatsapp/webhook → Signed Meta message ingestion.

GET /me → Own profile and server-derived staff flag.

GET/PATCH /profiles/{phone} → Own profile or authorised staff-assisted editing.

GET /profiles/{phone}/farms → Own farms or authorised staff lookup.

PATCH /profiles/{phone}/farms/{farm_id} → Edit farm details.

PUT /profiles/{phone}/weather → Opt in/out and set forecast preferences.

### Crop Yield Prediction
POST /predict → Predict crop yield from soil data and aerial images using prebuilt models

## 🔐 Security Notes

Always store secrets in .env, never in code.

Use HTTPS in production.

Rotate Meta & Azure keys regularly.

## 📌 Future Improvements

More granular staff roles and an agronomist approval workflow.

Offline-first support for rural deployments.

Integration with satellite NDVI imagery.

Advanced analytics dashboards.
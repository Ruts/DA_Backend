import hashlib
import json
import logging
import math
import os
import re
import secrets
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosResourceExistsError
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from meta_whatsapp import phone_id, required, valid_signature

UTC = timezone.utc
logger = logging.getLogger(__name__)


def utcnow():
    return datetime.now(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat()


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120, pattern=r"\S")
    language: Literal["en", "sw"] = "en"


class TakeoverUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paused: bool
    revision: str = Field(max_length=36)


class StaffMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    message: str = Field(min_length=1, max_length=4096, pattern=r"\S")
    requestId: UUID
    revision: str = Field(max_length=36)


def automation_paused(user):
    return (user or {}).get("active") is False or bool((user or {}).get("handoff", {}).get("paused"))


def staff_identity(actor):
    return actor["phone"] if actor.get("accountId") else phone_id(actor["phone"])


def automation_revision(user):
    return (user or {}).get("handoff", {}).get("revision", "")


def automation_blocked(user, job):
    handoff = (user or {}).get("handoff", {})
    return automation_paused(user) or (
        bool(handoff.get("changedAt")) and parse_time(job["createdAt"]) <= parse_time(handoff["changedAt"])
    )


class FarmUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=120, pattern=r"\S")
    crop: str = Field(min_length=1, max_length=80, pattern=r"\S")
    acreage: float = Field(gt=0, le=1000000)
    locationName: str = Field(default="", max_length=200)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    plantingDate: date | None = None

    @model_validator(mode="after")
    def paired_location(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Provide both latitude and longitude.")
        return self


class WeatherSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    consent: bool = False
    farmId: str = ""
    frequency: Literal["daily", "alternate", "weekly"] = "daily"
    time: str = Field(default="07:00", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    timezone: str = "Africa/Nairobi"
    weekday: int = Field(default=0, ge=0, le=6)

    @model_validator(mode="after")
    def valid_schedule(self):
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise ValueError("Choose a valid IANA timezone.")
        if self.enabled and (not self.consent or not self.farmId):
            raise ValueError("Select a farm and explicitly consent before enabling forecasts.")
        return self


def next_forecast(settings, after):
    zone = ZoneInfo(settings["timezone"])
    local = after.astimezone(zone)
    clock = time.fromisoformat(settings["time"])
    candidate = datetime.combine(local.date(), clock, zone)
    if settings["frequency"] == "weekly":
        candidate += timedelta(days=(settings["weekday"] - candidate.weekday()) % 7)
        if candidate <= local:
            candidate += timedelta(days=7)
    elif candidate <= local:
        candidate += timedelta(days=2 if settings["frequency"] == "alternate" else 1)
    return iso(candidate)


def public_profile(user, staff=False):
    fields = ("phoneNumber", "name", "language", "consentAt", "onboardingStep", "weather")
    return {
        **{key: user.get(key) for key in fields},
        "language": user.get("language") or "en", "isStaff": staff,
    }


def public_farm(farm):
    return {key: value for key, value in farm.items() if not key.startswith("_")}


PROMPTS = {
    "consent": (
        "Welcome to Digital Agronomist. We store your phone, farm details, messages and photos "
        "to provide farm advice using our AI provider, with authorised staff support. Farm coordinates "
        "are sent to the weather provider when you request forecasts. Advice is not a confirmed diagnosis. "
        "Weather messages are optional. Reply YES to consent, or NO to stop. "
        "Karibu. Tunahifadhi simu, taarifa za shamba, ujumbe na picha kwa ushauri na msaada wa "
        "wafanyakazi walioidhinishwa. Ushauri si utambuzi wa uhakika. Utabiri ni hiari. "
        "Jibu NDIYO kukubali au HAPANA kukataa."
    ),
    "language": ("Choose English or Kiswahili (EN / SW).", "Chagua English au Kiswahili (EN / SW)."),
    "declined": (
        "Registration paused. We will not send forecasts. Send START to see the consent request again.",
        "Usajili umesitishwa. Hatutatuma utabiri. Tuma START kuona ombi la idhini tena.",
    ),
    "name": ("What is your name?", "Unaitwa nani?"),
    "farm": ("What is your farm's name?", "Shamba lako linaitwaje?"),
    "location": (
        "Share your farm's location pin, or type latitude,longitude. You can SKIP and add it in the portal.",
        "Tuma eneo la shamba kwa pini au latitudo,longitudo. Andika SKIP kuongeza baadaye kwenye tovuti.",
    ),
    "crop": ("What crop are you growing?", "Unalima zao gani?"),
    "acreage": ("What is the farm area in acres?", "Shamba lina ukubwa wa ekari ngapi?"),
    "planting": (
        "When did you plant? Use YYYY-MM-DD or SKIP if unknown.",
        "Ulipanda lini? Tumia YYYY-MM-DD au SKIP ikiwa hujui.",
    ),
    "ready": (
        "Your account is ready. Ask a farm question or send a crop photo with a description. "
        "LOGIN: portal code; FARMS: list farms; FARM <number>: select; ADD FARM: another farm; "
        "WEATHER: forecast options; STOP: stop weather. AI advice can be uncertain; consult an agronomist.",
        "Akaunti iko tayari. Uliza swali au tuma picha na maelezo. LOGIN: msimbo wa tovuti; "
        "FARMS: mashamba; FARM <namba>: chagua; ADD FARM: ongeza; WEATHER: utabiri; STOP: simamisha utabiri. "
        "Ushauri wa AI unaweza kukosea; wasiliana na mtaalamu.",
    ),
}


class FarmerService:
    def __init__(self, users, farms, chats, jobs, messenger, adviser, interactions):
        self.users, self.farms, self.chats, self.jobs = users, farms, chats, jobs
        self.messenger, self.adviser = messenger, adviser
        self.interactions = interactions

    def record_inbound(self, message):
        kind = message["type"]
        details = message.get(kind, {})
        content = details.get("body") or details.get("caption") or f"[{kind}]"
        if kind == "location":
            content = f"Location: {details['latitude']}, {details['longitude']}"
        record = {
            "id": hashlib.sha256(f"in:{message['id']}".encode()).hexdigest(),
            "owner": phone_id(message["from"]), "direction": "inbound",
            "type": kind, "message": content, "timestamp": iso(
                datetime.fromtimestamp(int(message["timestamp"]), UTC) if message.get("timestamp") else utcnow()
            ),
            "status": "received", "metaMessageId": message["id"],
        }
        if kind == "image":
            record["mediaId"] = details["id"]
        try:
            self.interactions.create_item(record)
        except CosmosResourceExistsError:
            pass

    def send_recorded(self, phone, text, send, key=None):
        record = {
            "id": hashlib.sha256(f"out:{key}".encode()).hexdigest() if key else str(uuid4()),
            "owner": phone_id(phone), "direction": "outbound", "type": "text",
            "message": re.sub(r"(portal code: )\d{6}", r"\1[redacted]", text),
            "timestamp": iso(utcnow()), "status": "sending",
        }
        self.interactions.upsert_item(record)
        return self.complete_send(record, send)

    def complete_send(self, record, send):
        try:
            message_id = send()
        except Exception:
            record["status"] = "delivery_unknown"
            self.interactions.upsert_item(record)
            raise
        record.update(status="accepted", metaMessageId=message_id)
        self.interactions.upsert_item(record)
        return message_id

    def conversation(self, user):
        phone = phone_id(user["phoneNumber"])
        records = list(self.interactions.query_items(
            "SELECT TOP 1 * FROM c WHERE c.owner = @phone AND c.direction = 'inbound' "
            "ORDER BY c.timestamp DESC",
            parameters=[{"name": "@phone", "value": phone}], enable_cross_partition_query=True,
        ))
        times = [parse_time(value) for value in (
            user.get("lastInboundAt"), records[0]["timestamp"] if records else None,
        ) if value]
        now = utcnow()
        last_inbound = max(times) if times else None
        deadline = last_inbound + timedelta(hours=24) if last_inbound else None
        window_open = bool(last_inbound and last_inbound <= now < deadline)
        return {
            "phone": phone, "name": user.get("name", ""), "paused": automation_paused(user),
            "revision": automation_revision(user),
            "changedBy": user.get("handoff", {}).get("changedBy"),
            "changedAt": user.get("handoff", {}).get("changedAt"),
            "replyWindowEndsAt": iso(deadline) if deadline else None,
            "windowOpen": window_open,
        }

    def staff_message(self, user, values, actor):
        phone, staff_phone = phone_id(user["phoneNumber"]), staff_identity(actor)
        record_id = hashlib.sha256(f"staff:{staff_phone}:{phone}:{values.requestId}".encode()).hexdigest()

        def remember_reply(record):
            if record["status"] == "accepted" and record.get("farmId"):
                self.save_chat(
                    user, {"id": record["farmId"]}, "assistant",
                    "[Staff reply] " + record["message"], record_id,
                    staff_phone=staff_phone, timestamp=record["timestamp"],
                )

        def existing_send():
            records = list(self.interactions.query_items(
                "SELECT * FROM c WHERE c.id = @id",
                parameters=[{"name": "@id", "value": record_id}], enable_cross_partition_query=True,
            ))
            record = records[0]
            if record["message"] != values.message or record.get("revision") != values.revision:
                raise HTTPException(409, "This request ID was already used for a different message.")
            remember_reply(record)
            return {key: record[key] for key in ("id", "status")}

        existing = list(self.interactions.query_items(
            "SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": record_id}], enable_cross_partition_query=True,
        ))
        if existing:
            return existing_send()
        if not automation_paused(user) or automation_revision(user) != values.revision:
            raise HTTPException(409, "Take over this conversation and refresh before sending.")
        if not self.conversation(user)["windowOpen"]:
            raise HTTPException(
                409, "The 24-hour WhatsApp reply window is closed. Ask the farmer to message first; "
                "proactive contact outside this window requires an approved Meta template.",
            )
        for setting in ("META_GRAPH_VERSION", "META_ACCESS_TOKEN", "META_PHONE_NUMBER_ID"):
            if not os.getenv(setting):
                raise HTTPException(503, "WhatsApp is not configured. Contact the service administrator.")
        farm = self.active_farm(user)
        record = {
            "id": record_id, "owner": phone, "direction": "outbound", "type": "text",
            "senderType": "staff", "staffPhone": staff_phone, "revision": values.revision,
            "message": values.message, "timestamp": iso(utcnow()), "status": "sending",
            "farmId": farm["id"] if farm else None,
        }
        try:
            self.interactions.create_item(record)
        except CosmosResourceExistsError:
            return existing_send()
        # A concurrent resume must not release an old staff draft.
        latest = self.find_user(phone)
        if not automation_paused(latest) or automation_revision(latest) != values.revision:
            record["status"] = "cancelled"
            self.interactions.upsert_item(record)
            return {"id": record_id, "status": "cancelled"}
        try:
            self.complete_send(record, lambda: self.messenger.text(phone, values.message))
        except (RuntimeError, requests.RequestException) as error:
            logger.warning("Staff send outcome uncertain for interaction %s", record_id)
            raise HTTPException(
                502, "Send outcome is uncertain. Check the inbox and Meta before sending again.",
            ) from error
        remember_reply(record)
        return {"id": record_id, "status": record["status"]}

    def save_chat(self, user, farm, role, content, source_id, media_id=None, staff_phone=None, timestamp=None):
        record = {
            "id": hashlib.sha256(f"{source_id}:{role}".encode()).hexdigest(),
            "user": phone_id(user["phoneNumber"]), "farmId": farm["id"],
            "role": role, "message": content, "timestamp": timestamp or iso(utcnow()),
        }
        if media_id:
            record["mediaId"] = media_id
        if staff_phone:
            record.update(senderType="staff", staffPhone=staff_phone)
        self.chats.upsert_item(record)

    def find_user(self, phone):
        phone = phone_id(phone)
        # Older registrations used both +254... and 254... as Cosmos IDs.
        records = list(self.users.query_items(
            "SELECT * FROM c WHERE c.phoneNumber = @phone OR c.phoneNumber = @plus "
            "OR c.id = @phone OR c.id = @plus",
            parameters=[{"name": "@phone", "value": phone}, {"name": "@plus", "value": "+" + phone}],
            enable_cross_partition_query=True,
        ))
        if len(records) > 1:
            raise HTTPException(409, "Duplicate phone records need staff reconciliation.")
        return records[0] if records else None

    def staff(self, actor):
        if actor.get("accountId"):
            return actor.get("role") in {"admin", "staff"} and not actor.get("passwordChangeRequired")
        allowed = {phone_id(p) for p in os.getenv("STAFF_PHONES", "").split(",") if p.strip()}
        return phone_id(actor["phone"]) in allowed

    def authorised_user(self, actor, phone, include_inactive=False):
        try:
            target = phone_id(phone)
        except ValueError as error:
            raise HTTPException(422, str(error))
        if not self.staff(actor) and phone_id(actor["phone"]) != target:
            raise HTTPException(403, "You cannot access another farmer's records.")
        user = self.find_user(phone)
        if not user:
            raise HTTPException(404, "Farmer not found.")
        if user.get("active") is False and not include_inactive:
            raise HTTPException(403, "This farmer account is archived.")
        return user

    def save_user(self, user):
        saved = self.users.replace_item(
            item=user["id"], body=user, etag=user["_etag"],
            match_condition=MatchConditions.IfNotModified,
        )
        user.update(saved)
        return user

    def farm_list(self, user):
        phone = phone_id(user["phoneNumber"])
        return list(self.farms.query_items(
            "SELECT * FROM c WHERE c.owner = @phone OR c.owner = @plus ORDER BY c.id ASC",
            parameters=[{"name": "@phone", "value": phone}, {"name": "@plus", "value": "+" + phone}],
            enable_cross_partition_query=True,
        ))

    def farm(self, user, farm_id):
        farm = next((f for f in self.farm_list(user) if f["id"] == farm_id), None)
        if not farm:
            raise HTTPException(404, "Farm not found.")
        return farm

    def active_farm(self, user):
        farms = self.farm_list(user)
        return next((f for f in farms if f["id"] == user.get("activeFarmId")), farms[0] if farms else None)

    def update_weather(self, user, settings, actor):
        values = settings.model_dump()
        if settings.enabled:
            farm = self.farm(user, settings.farmId)
            if farm.get("latitude") is None or farm.get("longitude") is None:
                raise ValueError("Add the farm's location pin before enabling forecasts.")
        values["revision"] = str(uuid4())
        values["consentAt"] = iso(utcnow()) if settings.enabled else None
        values["updatedBy"] = actor
        values["nextDueAt"] = next_forecast(values, utcnow()) if settings.enabled else None
        user["weather"] = values
        return self.save_user(user)

    def issue_code(self, user, in_chat=False):
        if user.get("active") is False:
            raise HTTPException(403, "This farmer account is archived.")
        now = utcnow()
        if user.get("codeSentAt") and (now - parse_time(user["codeSentAt"])).total_seconds() < 60:
            raise HTTPException(429, "Wait one minute before requesting another code.")
        if not user.get("verified"):
            raise HTTPException(409, "Complete registration in WhatsApp first.")
        recent = user.get("lastInboundAt") and now - parse_time(user["lastInboundAt"]) < timedelta(hours=24)
        template = os.getenv("META_LOGIN_TEMPLATE")
        if not in_chat and not recent and not template:
            raise HTTPException(409, "Message LOGIN to Digital Agronomist on WhatsApp, then enter that code here.")
        code = str(secrets.randbelow(900000) + 100000)
        user["codeHash"] = self.code_hash(user, code)
        user["codeExpiresAt"] = iso(now + timedelta(minutes=10))
        user["codeSentAt"] = iso(now)
        user["codeAttempts"] = 0
        user.pop("code", None)
        self.save_user(user)
        text = f"Digital Agronomist portal code: {code}. Expires in 10 minutes. Do not share it."
        if in_chat:
            return text
        if recent:
            self.send_recorded(
                user["phoneNumber"], text, lambda: self.messenger.text(user["phoneNumber"], text),
            )
        else:
            self.send_recorded(
                user["phoneNumber"], "Portal login code: [redacted]",
                lambda: self.messenger.template(
                    user["phoneNumber"], template, os.getenv("META_LOGIN_LANGUAGE", "en_US"),
                    [code], authentication=True,
                ),
            )
        return None

    def code_hash(self, user, code):
        import hmac
        return hmac.new(
            required("TOKEN_SECRET_KEY").encode(),
            f"{phone_id(user['phoneNumber'])}:{code}".encode(), hashlib.sha256,
        ).hexdigest()

    def verify_code(self, user, code):
        import hmac
        if user.get("active") is False:
            raise HTTPException(403, "This farmer account is archived.")
        if (
            not user.get("codeHash") or user.get("codeAttempts", 0) >= 5
            or utcnow() >= parse_time(user["codeExpiresAt"])
        ):
            raise HTTPException(401, "Code expired or invalid. Request a new code.")
        user["codeAttempts"] = user.get("codeAttempts", 0) + 1
        if not hmac.compare_digest(user["codeHash"], self.code_hash(user, code)):
            self.save_user(user)
            raise HTTPException(401, "Invalid code.")
        user.pop("codeHash")
        self.save_user(user)

    def prompt(self, user, key):
        text = PROMPTS[key]
        return text if isinstance(text, str) else text[1 if user.get("language") == "sw" else 0]

    def onboard(self, user, message):
        text = message.get("text", {}).get("body", "").strip()
        step = user.get("onboardingStep", "consent")
        draft = user.setdefault("farmDraft", {})
        if step == "declined":
            if text.upper() in {"START", "ANZA"}:
                user["onboardingStep"] = "consent"
                return self.prompt(user, "consent")
            return self.prompt(user, "declined")
        if step == "consent":
            if text.upper() in {"NO", "HAPANA"}:
                user["onboardingStep"] = "declined"
                return self.prompt(user, "declined")
            if text.upper() not in {"YES", "NDIYO"}:
                return self.prompt(user, "consent")
            user["consentAt"] = iso(utcnow())
            step = "language"
        elif step == "language":
            if text.lower() not in {"en", "sw", "english", "kiswahili"}:
                return self.prompt(user, "language")
            user["language"] = "sw" if text.lower() in {"sw", "kiswahili"} else "en"
            step = "name"
        elif step == "name":
            if not text or len(text) > 120:
                return self.prompt(user, step)
            user["name"] = text
            step = "farm"
        elif step == "farm":
            if not text or len(text) > 120:
                return self.prompt(user, step)
            user["farmDraft"] = {"id": str(uuid4()), "name": text}
            step = "location"
        elif step == "location":
            location = message.get("location")
            if not location and text.upper() != "SKIP":
                try:
                    lat, lon = map(float, text.split(","))
                    location = {"latitude": lat, "longitude": lon}
                except ValueError:
                    return self.prompt(user, step)
            if location:
                lat, lon = location["latitude"], location["longitude"]
                if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    return self.prompt(user, step)
                draft.update(latitude=lat, longitude=lon, locationName=location.get("name", ""))
            step = "crop"
        elif step == "crop":
            if not text or len(text) > 80:
                return self.prompt(user, step)
            draft["crop"] = text
            step = "acreage"
        elif step == "acreage":
            try:
                area = float(text)
            except ValueError:
                return self.prompt(user, step)
            if not math.isfinite(area) or not 0 < area <= 1000000:
                return self.prompt(user, step)
            draft["acreage"] = area
            step = "planting"
        elif step == "planting":
            try:
                planted = date.fromisoformat(text).isoformat() if text.upper() != "SKIP" else None
            except ValueError:
                return self.prompt(user, step)
            farm = {
                **draft, "plantingDate": planted, "owner": phone_id(user["phoneNumber"]),
                "coordinates": (
                    [{"lat": draft["latitude"], "lng": draft["longitude"]}]
                    if "latitude" in draft else []
                ),
            }
            self.farms.upsert_item(farm)
            user["activeFarmId"] = farm["id"]
            user["verified"] = True
            user.pop("farmDraft", None)
            step = "ready"
        user["onboardingStep"] = step
        return self.prompt(user, step)

    def handle(self, user, message):
        text = message.get("text", {}).get("body", "").strip()
        command = text.upper()
        if command in {"STOP", "STOP WEATHER", "ACHA", "SIMAMISHA"}:
            self.update_weather(user, WeatherSettings(), user["phoneNumber"])
            return "Weather notifications stopped. / Ujumbe wa hali ya hewa umesimamishwa."
        if user.get("onboardingStep") != "ready":
            return self.onboard(user, message)
        if command == "LOGIN":
            return self.issue_code(user, in_chat=True)
        if command in {"HELP", "MENU", "MSAADA"}:
            return self.prompt(user, "ready")
        if command == "ADD FARM":
            user["onboardingStep"] = "farm"
            return self.prompt(user, "farm")
        if command == "FARMS":
            return "\n".join(f"{i}. {f['name']}" for i, f in enumerate(self.farm_list(user), 1)) + "\nFARM <number>"
        if command.startswith("FARM "):
            farms = self.farm_list(user)
            try:
                index = int(text.split()[1]) - 1
                if not 0 <= index < len(farms):
                    raise ValueError()
            except ValueError:
                return "Send FARMS, then FARM <number>."
            user["activeFarmId"] = farms[index]["id"]
            return f"Selected: {farms[index]['name']}"
        if command.startswith("WEATHER") or command.startswith("HALI"):
            return self.weather_command(user, text)
        farm = self.active_farm(user)
        if not farm:
            return "Send ADD FARM to register a farm first."
        if message["type"] not in {"text", "image"}:
            return "Please send a text question or crop photo. / Tuma swali au picha ya zao."
        question = text if message["type"] == "text" else message["image"].get("caption", "")
        question = question or "What can you observe in this crop photo? What further information is needed?"
        history = list(self.chats.query_items(
            "SELECT TOP 10 * FROM c WHERE c.user = @phone AND c.farmId = @farm ORDER BY c.timestamp DESC",
            parameters=[
                {"name": "@phone", "value": phone_id(user["phoneNumber"])},
                {"name": "@farm", "value": farm["id"]},
            ], enable_cross_partition_query=True,
        ))
        image = self.messenger.image(message["image"]["id"]) if message["type"] == "image" else None
        answer = self.adviser(user, farm, list(reversed(history)), question, image)
        for role, content in (("user", question), ("assistant", answer)):
            self.save_chat(
                user, farm, role, content, message["id"],
                media_id=message["image"]["id"] if image and role == "user" else None,
            )
        return answer

    def weather_command(self, user, text):
        parts = text.upper().split()
        if len(parts) == 1:
            return (
                "Forecasts are optional. To consent and enable for the selected farm: "
                "WEATHER DAILY 07:00, WEATHER ALTERNATE 07:00, or WEATHER WEEKLY MON 07:00. "
                "Timezone: Africa/Nairobi (editable in portal). WEATHER NOW: current forecast; STOP: unsubscribe. "
                "/ Utabiri ni hiari. Tumia amri hizi kukubali; STOP kusimamisha."
            )
        farm = self.active_farm(user)
        if not farm:
            return "Send ADD FARM first."
        if parts[1] == "NOW":
            return self.forecast(farm, user.get("language", "en"))
        days = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        try:
            frequency = {"DAILY": "daily", "ALTERNATE": "alternate", "WEEKLY": "weekly"}[parts[1]]
            weekday = days.index(parts[2]) if frequency == "weekly" else 0
            clock = parts[3] if frequency == "weekly" else parts[2]
            if len(parts) != (4 if frequency == "weekly" else 3):
                raise ValueError("Unexpected extra arguments.")
            settings = WeatherSettings(
                enabled=True, consent=True, farmId=farm["id"], frequency=frequency,
                time=clock, weekday=weekday,
                timezone=user.get("weather", {}).get("timezone", "Africa/Nairobi"),
            )
            updated = self.update_weather(user, settings, user["phoneNumber"])
            user.update(updated)
        except (KeyError, IndexError, ValueError):
            return "Check farm coordinates and use WEATHER DAILY 07:00, WEATHER ALTERNATE 07:00 or WEATHER WEEKLY MON 07:00."
        return f"Weather enabled: {frequency}, {clock}, {settings.timezone}. STOP to unsubscribe."

    def forecast(self, farm, language="en"):
        if farm.get("latitude") is None or farm.get("longitude") is None:
            raise ValueError("Add this farm's location in the portal before requesting weather.")
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": farm["latitude"], "longitude": farm["longitude"],
                "daily": "temperature_2m_min,temperature_2m_max,precipitation_probability_max,precipitation_sum",
                "timezone": "Africa/Nairobi", "forecast_days": 3,
            }, timeout=25,
        )
        response.raise_for_status()
        daily = response.json()["daily"]
        lines = [farm["name"]]
        for i, day in enumerate(daily["time"]):
            low, high = daily["temperature_2m_min"][i], daily["temperature_2m_max"][i]
            probability, rain = daily["precipitation_probability_max"][i], daily["precipitation_sum"][i]
            if any(value is None for value in (low, high, probability, rain)):
                raise ValueError("Weather provider returned incomplete readings; try again later.")
            label = "Mvua" if language == "sw" else "Rain"
            lines.append(f"{day}: {low}-{high} C; {label} {probability}%, {rain} mm.")
        lines.append(
            "Chanzo: Open-Meteo. Utabiri unaweza kubadilika. STOP: simamisha."
            if language == "sw" else "Source: Open-Meteo. Forecasts can change. STOP to unsubscribe."
        )
        return "\n".join(lines)

    def enqueue(self, job):
        try:
            self.jobs.create_item(job)
        except CosmosResourceExistsError:
            return False
        return True


def create_router(service, current_user):
    router = APIRouter()

    @router.get("/service-info")
    def service_info():
        number = phone_id(required("WHATSAPP_BUSINESS_NUMBER"))
        return {"whatsappNumber": number, "whatsappUrl": f"https://wa.me/{number}"}

    @router.get("/whatsapp/webhook", response_class=PlainTextResponse)
    def verify_webhook(request: Request):
        query = request.query_params
        if (
            query.get("hub.mode") != "subscribe"
            or not secrets.compare_digest(query.get("hub.verify_token", ""), required("META_VERIFY_TOKEN"))
        ):
            raise HTTPException(403, "Webhook verification failed.")
        return query.get("hub.challenge", "")

    @router.post("/whatsapp/webhook")
    async def receive(request: Request):
        body = await request.body()
        if len(body) > 1024 * 1024:
            raise HTTPException(413, "Webhook payload too large.")
        if not valid_signature(body, request.headers.get("x-hub-signature-256")):
            raise HTTPException(403, "Invalid webhook signature.")
        try:
            payload = json.loads(body)
        except ValueError:
            raise HTTPException(400, "Invalid JSON.")
        if not isinstance(payload, dict) or payload.get("object") != "whatsapp_business_account":
            raise HTTPException(400, "Unexpected webhook object.")
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if value.get("metadata", {}).get("phone_number_id") != required("META_PHONE_NUMBER_ID"):
                    continue
                for message in value.get("messages", []):
                    phone = phone_id(message["from"])
                    service.record_inbound(message)
                    service.enqueue({
                        "id": hashlib.sha256(message["id"].encode()).hexdigest(), "owner": phone,
                        "kind": "inbound", "message": message, "status": "pending",
                        "createdAt": iso(utcnow()), "availableAt": iso(utcnow()),
                    })
                for status in value.get("statuses", []):
                    if status.get("status") == "failed":
                        logger.error("Meta reported failed delivery for message %s", status.get("id"))
        return {"status": "accepted"}

    @router.get("/staff/interactions")
    def interactions(
        phone: str | None = None,
        source: Literal["whatsapp", "advice"] = "whatsapp",
        cursor: str | None = Query(default=None, max_length=16384),
        limit: int = Query(default=50, ge=1, le=100),
        actor=Depends(current_user),
    ):
        if not service.staff(actor):
            raise HTTPException(403, "Staff access required.")
        parameters = []
        query = "SELECT * FROM c"
        if phone:
            try:
                target = phone_id(phone)
            except ValueError as error:
                raise HTTPException(422, str(error))
            field = "owner" if source == "whatsapp" else "user"
            query += f" WHERE c.{field} = @phone OR c.{field} = @plus"
            parameters = [{"name": "@phone", "value": target}, {"name": "@plus", "value": "+" + target}]
        query += " ORDER BY c.timestamp DESC"
        container = service.interactions if source == "whatsapp" else service.chats
        pages = container.query_items(
            query, parameters=parameters, enable_cross_partition_query=True, max_item_count=limit,
        ).by_page(continuation_token=cursor)
        page = next(pages, [])
        records = []
        for item in page:
            if source == "advice":
                records.append({
                    "id": item["id"], "owner": item.get("user"),
                    "direction": "inbound" if item.get("role") == "user" else "outbound",
                    "type": "image" if item.get("mediaId") else "text",
                    "message": item.get("message", ""), "timestamp": item.get("timestamp"),
                    "status": "stored", "mediaId": item.get("mediaId"),
                    "senderType": item.get("senderType"), "staffPhone": item.get("staffPhone"),
                })
            else:
                records.append({key: value for key, value in item.items() if not key.startswith("_")})
        return {"items": records, "cursor": pages.continuation_token}

    def staff_user(actor, phone):
        if not service.staff(actor):
            raise HTTPException(403, "Staff access required.")
        return service.authorised_user(actor, phone)

    @router.get("/staff/conversations/{phone}")
    def conversation(phone: str, actor=Depends(current_user)):
        return service.conversation(staff_user(actor, phone))

    @router.put("/staff/conversations/{phone}")
    def takeover(phone: str, values: TakeoverUpdate, actor=Depends(current_user)):
        user = staff_user(actor, phone)
        if automation_revision(user) != values.revision:
            raise HTTPException(409, "Conversation controls changed. Refresh before trying again.")
        if automation_paused(user) == values.paused:
            return service.conversation(user)
        handoff = {
            "paused": values.paused, "revision": str(uuid4()),
            "changedBy": staff_identity(actor), "changedAt": iso(utcnow()),
        }
        user["handoff"] = handoff
        service.save_user(user)
        service.interactions.create_item({
            "id": handoff["revision"], "owner": phone_id(user["phoneNumber"]), "direction": "internal",
            "type": "handoff", "senderType": "staff", "staffPhone": handoff["changedBy"],
            "timestamp": handoff["changedAt"], "status": "recorded",
            "message": (
                "Staff took over. AI replies and weather notifications paused."
                if values.paused else "Staff resumed AI replies and opted-in weather notifications."
            ),
        })
        return service.conversation(user)

    @router.post("/staff/conversations/{phone}/messages")
    def staff_message(phone: str, values: StaffMessage, actor=Depends(current_user)):
        return service.staff_message(staff_user(actor, phone), values, actor)

    @router.get("/staff/interactions/{interaction_id}/image")
    def interaction_image(
        interaction_id: str, source: Literal["whatsapp", "advice"] = "whatsapp",
        actor=Depends(current_user),
    ):
        if not service.staff(actor):
            raise HTTPException(403, "Staff access required.")
        container = service.interactions if source == "whatsapp" else service.chats
        records = list(container.query_items(
            "SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": interaction_id}], enable_cross_partition_query=True,
        ))
        if not records or not records[0].get("mediaId"):
            raise HTTPException(404, "Photo not found.")
        try:
            data, mime = service.messenger.image(records[0]["mediaId"])
        except (ValueError, RuntimeError, requests.RequestException) as error:
            logger.warning("Staff photo retrieval failed for interaction %s", interaction_id)
            raise HTTPException(502, "Photo unavailable from Meta; older media may have expired.") from error
        return Response(data, media_type=mime, headers={
            "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
        })

    @router.get("/me")
    def me(actor=Depends(current_user)):
        user = service.authorised_user(actor, actor["phone"])
        return public_profile(user, service.staff(actor))

    @router.get("/profiles/{phone}")
    def profile(phone: str, actor=Depends(current_user)):
        return public_profile(service.authorised_user(actor, phone))

    @router.patch("/profiles/{phone}")
    def update_profile(phone: str, values: ProfileUpdate, actor=Depends(current_user)):
        user = service.authorised_user(actor, phone)
        user.update(values.model_dump())
        user["updatedBy"] = staff_identity(actor)
        return public_profile(service.save_user(user))

    @router.get("/profiles/{phone}/farms")
    def farms(phone: str, actor=Depends(current_user)):
        return [public_farm(f) for f in service.farm_list(service.authorised_user(actor, phone))]

    @router.patch("/profiles/{phone}/farms/{farm_id}")
    def update_farm(phone: str, farm_id: str, values: FarmUpdate, actor=Depends(current_user)):
        user = service.authorised_user(actor, phone)
        farm = service.farm(user, farm_id)
        farm.update(values.model_dump(mode="json"))
        farm["updatedBy"] = staff_identity(actor)
        service.farms.replace_item(
            item=farm["id"], body=farm, etag=farm["_etag"],
            match_condition=MatchConditions.IfNotModified,
        )
        return public_farm(farm)

    @router.put("/profiles/{phone}/weather")
    def weather(phone: str, values: WeatherSettings, actor=Depends(current_user)):
        user = service.authorised_user(actor, phone)
        try:
            user = service.update_weather(user, values, actor["phone"])
        except ValueError as error:
            raise HTTPException(422, str(error))
        return user["weather"]

    return router

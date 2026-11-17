import copy
import hashlib
import hmac
import importlib.util
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from azure.cosmos.exceptions import CosmosAccessConditionFailedError, CosmosResourceExistsError
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from farmer_service import (
    FarmerService, FarmUpdate, WeatherSettings, create_router, iso, next_forecast, utcnow,
)
from meta_whatsapp import MetaWhatsApp
from whatsapp_worker import WhatsAppWorker


class Pages:
    def __init__(self, items, limit, cursor):
        self.items, self.limit = items, limit
        self.offset = int(cursor or 0)
        self.continuation_token = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.offset >= len(self.items):
            raise StopIteration
        page = self.items[self.offset:self.offset + self.limit]
        self.offset += len(page)
        self.continuation_token = str(self.offset) if self.offset < len(self.items) else None
        return page


class Results(list):
    def __init__(self, items, limit):
        super().__init__(items)
        self.limit = limit

    def by_page(self, continuation_token=None):
        return Pages(self, self.limit, continuation_token)


class Container:
    def __init__(self):
        self.items = {}
        self.revision = 0

    def upsert_item(self, body):
        self.revision += 1
        item = copy.deepcopy(body)
        item["_etag"] = str(self.revision)
        self.items[item["id"]] = item
        return copy.deepcopy(item)

    def create_item(self, body):
        if body["id"] in self.items:
            raise CosmosResourceExistsError(message="duplicate")
        return self.upsert_item(body)

    def replace_item(self, item, body, etag, match_condition):
        if self.items[item]["_etag"] != etag:
            raise CosmosAccessConditionFailedError(message="stale")
        return self.upsert_item(body)

    def query_items(self, query, parameters=None, **kwargs):
        values = {p["name"]: p["value"] for p in parameters or []}
        items = list(self.items.values())
        if "@id" in values:
            items = [i for i in items if i["id"] == values["@id"]]
        if "@phone" in values:
            choices = {values["@phone"], values.get("@plus", values["@phone"])}
            if "c.phoneNumber" in query:
                items = [i for i in items if i.get("phoneNumber") in choices or i["id"] in choices]
            elif "c.owner" in query:
                items = [i for i in items if i.get("owner") in choices]
            else:
                items = [i for i in items if i.get("user") in choices]
        if "@farm" in values:
            items = [i for i in items if i.get("farmId") == values["@farm"]]
        if "@owner" in values:
            items = [
                i for i in items if i["owner"] == values["@owner"] and i["kind"] == "inbound"
                and i["createdAt"] < values["@created"]
                and i["status"] in {"pending", "processing", "failed"}
            ]
        if "c.weather.enabled" in query:
            items = [
                i for i in items if i.get("weather", {}).get("enabled")
                and i["weather"]["nextDueAt"] <= values["@now"]
            ]
        if "c.status = 'pending'" in query and "@now" in values:
            items = [
                i for i in items if (
                    i["status"] == "pending" and i["availableAt"] <= values["@now"]
                    or i["status"] == "processing" and i["leaseUntil"] <= values["@now"]
                )
            ]
        if "ORDER BY c.createdAt" in query:
            items.sort(key=lambda i: i["createdAt"])
        if "ORDER BY c.timestamp" in query:
            items.sort(key=lambda i: i["timestamp"], reverse="DESC" in query)
        if "TOP 10" in query:
            items = items[:10]
        if "TOP 50" in query:
            items = items[:50]
        return Results(copy.deepcopy(items), kwargs.get("max_item_count", 100))


class FarmerTests(unittest.TestCase):
    def setUp(self):
        self.settings = patch.dict(os.environ, {
            "META_APP_SECRET": "test-app-secret",
            "META_VERIFY_TOKEN": "test-verify-token",
            "META_PHONE_NUMBER_ID": "test-phone-id",
            "META_ACCESS_TOKEN": "test-token",
            "META_GRAPH_VERSION": "v23.0",
            "TOKEN_SECRET_KEY": "test-jwt-secret",
            "META_WEATHER_TEMPLATE_EN": "farm_weather",
            "META_WEATHER_TEMPLATE_SW": "farm_weather",
            "STAFF_PHONES": "254700000009",
            "WHATSAPP_BUSINESS_NUMBER": "254700000099",
        })
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.users, self.farms, self.chats, self.jobs = (Container() for _ in range(4))
        self.interactions = Container()
        self.sender = Mock()
        self.sender.text.return_value = "outbound-id"
        self.sender.template.return_value = "weather-id"
        self.sender.image.return_value = (b"photo", "image/jpeg")
        self.adviser = Mock(return_value="Please share a clearer close-up and consult an agronomist.")
        self.service = FarmerService(
            self.users, self.farms, self.chats, self.jobs, self.sender, self.adviser, self.interactions,
        )
        self.user = self.users.create_item({
            "id": "254700000001", "phoneNumber": "254700000001", "name": "Farmer",
            "verified": True, "language": "en", "onboardingStep": "ready", "activeFarmId": "farm1",
        })
        self.farm = self.farms.create_item({
            "id": "farm1", "owner": self.user["phoneNumber"], "name": "North",
            "crop": "Maize", "acreage": 2, "latitude": -1.2, "longitude": 36.8,
        })
        self.actor = {"phone": self.user["phoneNumber"]}
        self.app = FastAPI()
        self.app.include_router(create_router(self.service, lambda: self.actor))
        self.client = TestClient(self.app)

    def message(self, text="", kind="text", number=1, **fields):
        return {
            "id": f"inbound-{number}", "from": self.user["phoneNumber"],
            "type": kind, "text": {"body": text}, **fields,
        }

    def job(self, message):
        return self.jobs.create_item({
            "id": message["id"], "owner": message["from"], "kind": "inbound",
            "message": message, "status": "pending", "availableAt": iso(utcnow()),
            "createdAt": iso(utcnow()),
        })

    def test_staff_inbox_paginates_filters_and_denies_farmers(self):
        self.service.record_inbound(self.message("Hello"))
        self.service.record_inbound({**self.message("Second farmer", number=2), "from": "254700000002"})
        self.assertEqual(self.client.get("/staff/interactions").status_code, 403)
        self.actor["phone"] = "254700000009"
        first = self.client.get("/staff/interactions?limit=1").json()
        self.assertEqual(len(first["items"]), 1)
        self.assertTrue(first["cursor"])
        second = self.client.get("/staff/interactions", params={
            "limit": 1, "cursor": first["cursor"],
        }).json()
        self.assertNotEqual(first["items"][0]["id"], second["items"][0]["id"])
        filtered = self.client.get("/staff/interactions", params={"phone": "+254700000001"}).json()
        self.assertEqual(len(filtered["items"]), 1)
        self.assertEqual(filtered["items"][0]["message"], "Hello")
        self.assertNotIn("_etag", filtered["items"][0])

    def test_full_whatsapp_exchange_is_archived_without_login_codes(self):
        self.job(self.message("LOGIN"))
        WhatsAppWorker(self.service).tick()
        records = list(self.interactions.items.values())
        self.assertEqual(len(records), 2)
        outbound = next(record for record in records if record["direction"] == "outbound")
        self.assertEqual(outbound["status"], "accepted")
        self.assertIn("[redacted]", outbound["message"])
        self.assertNotRegex(outbound["message"], r"\d{6}")

    def test_inbox_photo_proxy_is_staff_only_and_duplicate_receipts_are_idempotent(self):
        message = self.message(kind="image", image={"id": "photo-id", "caption": "Yellow leaves"})
        self.service.record_inbound(message)
        self.service.record_inbound(message)
        self.assertEqual(len(self.interactions.items), 1)
        record = next(iter(self.interactions.items.values()))
        path = f"/staff/interactions/{record['id']}/image"
        self.assertEqual(self.client.get(path).status_code, 403)
        self.sender.image.assert_not_called()
        self.actor["phone"] = "254700000009"
        result = self.client.get(path)
        self.assertEqual(result.content, b"photo")
        self.assertEqual(result.headers["cache-control"], "no-store")

    def test_weather_archive_keeps_forecast_and_uncertain_send_is_labelled(self):
        self.service.send_recorded("254700000001", "Rain likely tomorrow", lambda: "weather-id")
        self.sender.text.side_effect = TimeoutError("Uncertain send")
        with self.assertRaises(TimeoutError):
            self.service.send_recorded(
                "254700000001", "Hello", lambda: self.sender.text("254700000001", "Hello"),
            )
        self.assertEqual(
            {item["status"] for item in self.interactions.items.values()},
            {"accepted", "delivery_unknown"},
        )

    def test_registration_is_guided_and_creates_shared_records(self):
        user = self.users.create_item({
            "id": "254700000002", "phoneNumber": "254700000002", "language": "en",
            "onboardingStep": "consent", "name": "", "verified": False,
        })
        self.assertIn("consent", self.service.onboard(user, self.message("Hello")))
        for value in ("YES", "SW", "Amina", "South"):
            self.service.onboard(user, self.message(value))
        self.service.onboard(user, self.message(kind="location", location={"latitude": 0.1, "longitude": 35.0}))
        for value in ("Beans", "3.5", "2026-11-01"):
            self.service.onboard(user, self.message(value))
        self.service.save_user(user)
        restored = self.service.find_user("+254700000002")
        farm = self.service.farm_list(restored)[0]
        self.assertTrue(restored["verified"])
        self.assertEqual(restored["language"], "sw")
        self.assertEqual(farm["crop"], "Beans")
        self.assertEqual(farm["acreage"], 3.5)
        self.assertEqual(farm["plantingDate"], "2026-11-01")
        self.assertNotIn("weather", restored)

    def test_invalid_area_and_location_do_not_advance(self):
        for text in ("nan", "-2", "0", "infinity"):
            user = {**self.user, "onboardingStep": "acreage", "farmDraft": {}}
            self.service.onboard(user, self.message(text))
            self.assertEqual(user["onboardingStep"], "acreage")
        user = {**self.user, "onboardingStep": "location", "farmDraft": {}}
        self.service.onboard(user, self.message("91,36"))
        self.assertEqual(user["onboardingStep"], "location")
        with self.assertRaises(ValueError):
            FarmUpdate(name="North", crop="Maize", acreage=1, latitude=1)

    def test_declined_registration_never_calls_ai(self):
        user = {**self.user, "verified": False, "onboardingStep": "consent"}
        self.assertIn("paused", self.service.handle(user, self.message("NO")))
        self.service.handle(user, self.message("What crop is this?"))
        self.assertEqual(user["onboardingStep"], "declined")
        self.adviser.assert_not_called()
        self.service.handle(user, self.message("START"))
        self.assertEqual(user["onboardingStep"], "consent")

    def test_photo_advice_is_scoped_to_selected_farm(self):
        self.chats.create_item({
            "id": "other", "user": "254700000002", "farmId": "farm1",
            "message": "private other history", "role": "user", "timestamp": iso(utcnow()),
        })
        self.chats.create_item({
            "id": "old", "user": self.user["phoneNumber"], "farmId": "farm1",
            "message": "yellow leaves", "role": "user", "timestamp": iso(utcnow()),
        })
        reply = self.service.handle(self.user, self.message(
            kind="image", image={"id": "photo-id", "caption": "What is wrong?"}
        ))
        self.assertIn("agronomist", reply)
        args = self.adviser.call_args.args
        self.assertEqual(args[1]["id"], "farm1")
        self.assertEqual([m["id"] for m in args[2]], ["old"])
        self.assertEqual(args[3:], ("What is wrong?", (b"photo", "image/jpeg")))
        self.assertEqual(len(self.chats.items), 4)

    def test_farmer_cannot_edit_other_records_or_promote_self(self):
        self.assertEqual(self.client.get("/profiles/254700000002").status_code, 403)
        response = self.client.patch("/profiles/254700000001", json={
            "name": "Farmer", "language": "en", "isStaff": True,
        })
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.patch("/profiles/254700000001/farms/missing", json={
            "name": "Other", "crop": "Maize", "acreage": 2,
        }).status_code, 404)

    def test_authorised_staff_can_edit_without_changing_owner(self):
        self.actor["phone"] = "254700000009"
        response = self.client.patch("/profiles/254700000001/farms/farm1", json={
            "name": "Updated", "crop": "Beans", "acreage": 3,
            "latitude": -1, "longitude": 36, "plantingDate": "2026-11-01",
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.farms.items["farm1"]["owner"], self.user["phoneNumber"])
        self.assertEqual(self.farms.items["farm1"]["updatedBy"], self.actor["phone"])
        result = self.client.patch("/profiles/254700000001", json={"name": "New", "language": "sw"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.service.find_user(self.user["phoneNumber"])["name"], "New")

    def test_weather_requires_consent_and_real_farm_coordinates(self):
        body = {"enabled": True, "consent": False, "farmId": "farm1"}
        self.assertEqual(self.client.put("/profiles/254700000001/weather", json=body).status_code, 422)
        farm = self.farms.items["farm1"]
        farm.pop("latitude")
        body["consent"] = True
        self.assertEqual(self.client.put("/profiles/254700000001/weather", json=body).status_code, 422)
        body["timezone"] = "Not/AZone"
        self.assertEqual(self.client.put("/profiles/254700000001/weather", json=body).status_code, 422)

    def test_schedule_boundaries_and_frequencies(self):
        base = WeatherSettings(enabled=True, consent=True, farmId="farm1").model_dump()
        now = datetime(2026, 11, 16, 4, 0, tzinfo=timezone.utc)  # Monday 07:00 Nairobi
        self.assertEqual(next_forecast(base, now), "2026-11-17T04:00:00+00:00")
        base["frequency"] = "alternate"
        self.assertEqual(next_forecast(base, now), "2026-11-18T04:00:00+00:00")
        base["frequency"] = "weekly"
        self.assertEqual(next_forecast(base, now), "2026-11-23T04:00:00+00:00")
        base["weekday"] = 2
        self.assertEqual(next_forecast(base, now), "2026-11-18T04:00:00+00:00")

    def test_weather_commands_change_frequency_and_stop(self):
        for command, expected in (
            ("WEATHER DAILY 07:00", "daily"), ("WEATHER ALTERNATE 08:00", "alternate"),
            ("WEATHER WEEKLY FRI 09:00", "weekly"),
        ):
            self.assertIn("enabled", self.service.handle(self.user, self.message(command)))
            self.assertEqual(self.user["weather"]["frequency"], expected)
        self.service.handle(self.user, self.message("STOP"))
        restored = self.service.find_user(self.user["phoneNumber"])
        self.assertFalse(restored["weather"]["enabled"])
        self.assertIsNone(restored["weather"]["nextDueAt"])

    def test_signed_webhook_is_durable_and_deduplicated(self):
        payload = {
            "object": "whatsapp_business_account", "entry": [{"changes": [{"value": {
                "metadata": {"phone_number_id": "test-phone-id"},
                "messages": [self.message("Hello")],
            }}]}],
        }
        body = json.dumps(payload).encode()
        url = "/whatsapp/webhook"
        self.assertEqual(self.client.post(url, content=body).status_code, 403)
        signature = "sha256=" + hmac.new(b"test-app-secret", body, hashlib.sha256).hexdigest()
        for _ in range(2):
            response = self.client.post(url, content=body, headers={"x-hub-signature-256": signature})
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.jobs.items), 1)
        self.sender.text.assert_not_called()
        self.assertEqual(self.client.get(url, params={
            "hub.mode": "subscribe", "hub.verify_token": "test-verify-token", "hub.challenge": "123",
        }).text, "123")

    def test_worker_sends_once_and_resumes_after_restart(self):
        self.job(self.message("Why are leaves yellow?"))
        worker = WhatsAppWorker(self.service)
        worker.tick()
        WhatsAppWorker(self.service).tick()
        self.sender.text.assert_called_once()
        self.assertEqual(self.jobs.items["inbound-1"]["status"], "sent")

    def test_unknown_delivery_is_not_automatically_retried(self):
        self.job(self.message("HELP"))
        self.sender.text.side_effect = TimeoutError("simulated network uncertainty")
        worker = WhatsAppWorker(self.service)
        with self.assertLogs("whatsapp_worker", level="ERROR"):
            worker.tick()
        self.assertEqual(self.jobs.items["inbound-1"]["status"], "delivery_unknown")
        worker.tick()
        self.sender.text.assert_called_once()

    def test_retrying_message_blocks_later_conversation_steps(self):
        first = self.job(self.message("Why are leaves yellow?"))
        second = self.job(self.message("More details", number=2))
        self.adviser.side_effect = RuntimeError("AI unavailable")
        worker = WhatsAppWorker(self.service)
        with self.assertLogs("whatsapp_worker", level="ERROR"):
            worker.tick()
        self.assertEqual(self.jobs.items[first["id"]]["status"], "pending")
        self.assertEqual(self.jobs.items[second["id"]]["status"], "pending")
        self.assertNotIn("attempts", self.jobs.items[second["id"]])

    def test_blocked_backlog_does_not_starve_other_farmers(self):
        first = self.job(self.message("Waiting for retry"))
        first["availableAt"] = iso(utcnow() + timedelta(days=1))
        self.jobs.upsert_item(first)
        for number in range(2, 55):
            self.job(self.message("Follow-up", number=number))
        other = self.job({**self.message("Hello", number=100), "from": "254700000002"})
        WhatsAppWorker(self.service).tick()
        self.assertEqual(self.jobs.items[other["id"]]["status"], "sent")
        self.sender.text.assert_called_once()

    def test_legacy_profile_can_be_saved_and_invalid_lookup_is_rejected(self):
        self.user.pop("language")
        self.service.save_user(self.user)
        profile = self.client.get(f"/profiles/{self.user['phoneNumber']}").json()
        self.assertEqual(profile["language"], "en")
        response = self.client.patch(f"/profiles/{self.user['phoneNumber']}", json={
            "name": profile["name"], "language": profile["language"],
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/profiles/not-a-phone").status_code, 422)

    def test_queued_forecast_is_cancelled_after_optout(self):
        self.service.update_weather(self.user, WeatherSettings(
            enabled=True, consent=True, farmId="farm1",
        ), self.user["phoneNumber"])
        self.user["weather"]["nextDueAt"] = iso(utcnow() - timedelta(minutes=1))
        self.service.save_user(self.user)
        worker = WhatsAppWorker(self.service)
        worker.schedule()
        worker.schedule()
        self.assertEqual(len(self.jobs.items), 1)
        user = self.service.find_user(self.user["phoneNumber"])
        self.service.handle(user, self.message("STOP"))
        worker.tick()
        self.sender.template.assert_not_called()
        self.assertEqual(next(iter(self.jobs.items.values()))["status"], "cancelled")

    def test_scheduled_forecast_uses_approved_template(self):
        self.service.update_weather(self.user, WeatherSettings(
            enabled=True, consent=True, farmId="farm1", frequency="alternate",
        ), self.user["phoneNumber"])
        self.user["weather"]["nextDueAt"] = iso(utcnow() - timedelta(minutes=1))
        self.service.save_user(self.user)
        with patch.object(self.service, "forecast", return_value="North\nRain 70%"):
            WhatsAppWorker(self.service).tick()
        self.sender.template.assert_called_once_with(
            self.user["phoneNumber"], "farm_weather", "en", ["North Rain 70%"],
        )
        self.sender.text.assert_not_called()

    def test_login_code_expires_is_single_use_and_not_in_profile(self):
        code_text = self.service.issue_code(self.user, in_chat=True)
        code = code_text.split(": ")[1].split(".")[0]
        profile = self.client.get("/me").json()
        self.assertNotIn("codeHash", profile)
        with self.assertRaises(HTTPException):
            self.service.verify_code(self.user, "wrong")
        self.service.verify_code(self.user, code)
        with self.assertRaises(HTTPException):
            self.service.verify_code(self.user, code)
        user = self.service.find_user(self.user["phoneNumber"])
        user["codeSentAt"] = iso(utcnow() - timedelta(minutes=11))
        self.service.save_user(user)
        code = self.service.issue_code(user, in_chat=True).split(": ")[1].split(".")[0]
        user["codeExpiresAt"] = iso(utcnow() - timedelta(seconds=1))
        with self.assertRaises(HTTPException):
            self.service.verify_code(user, code)

    def test_meta_media_uses_authorization_and_rejects_external_host(self):
        sender = MetaWhatsApp()
        with patch.object(sender, "graph", return_value={
            "mime_type": "image/jpeg", "file_size": 30, "url": "https://example.com/image.jpg",
        }), patch("meta_whatsapp.requests.get") as get:
            with self.assertRaises(ValueError):
                sender.image("photo")
            get.assert_not_called()

    def test_api_login_and_profile_roundtrip_without_ml_or_live_services(self):
        cosmos = Mock()
        containers = {
            "users": self.users, "farms": self.farms, "chats": self.chats, "whatsapp_jobs": self.jobs,
        }
        cosmos.get_database_client.return_value.get_container_client.side_effect = (
            lambda name: containers.setdefault(name, Container())
        )
        self.user["lastInboundAt"] = iso(utcnow())
        self.service.save_user(self.user)
        with patch.dict(os.environ, {
            "AZURE_ENVIRONMENT": "production", "GOOGLE_API_KEY": "test-key",
            "TOKEN_ALGORITHM": "HS256", "COSMOS_ENDPOINT": "https://example.invalid",
            "COSMOS_KEY": "test-key", "DATABASE_NAME": "test-db",
            "AZURE_STORAGE_CONNECTION_STRING": "test-storage",
        }), patch("azure.cosmos.CosmosClient", return_value=cosmos), patch(
            "azure.storage.blob.BlobServiceClient.from_connection_string"
        ):
            spec = importlib.util.spec_from_file_location(
                "digital_agronomist_test_app", Path(__file__).with_name("main.py"),
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.farmer_service.messenger = self.sender
            client = TestClient(module.app)
            self.assertEqual(client.get("/me").status_code, 401)
            self.assertEqual(client.get("/me", headers={"Authorization": "Bearer invalid"}).status_code, 401)
            login = client.post("/login", json={"phone": "+254700000001"})
            self.assertEqual(login.status_code, 200, login.text)
            code = self.sender.text.call_args.args[1].split(": ")[1].split(".")[0]
            verified = client.post("/verify-code", json={"phone": "+254700000001", "code": code})
            self.assertEqual(verified.status_code, 200, verified.text)
            headers = {"Authorization": f"Bearer {verified.json()['token']}"}
            result = client.get("/me", headers=headers)
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["name"], "Farmer")
            self.assertEqual(client.get("/farms", headers=headers).json()[0]["id"], "farm1")
            self.assertEqual(client.post("/verify-code", json={
                "phone": "+254700000001", "code": code,
            }).status_code, 401)


if __name__ == "__main__":
    unittest.main()

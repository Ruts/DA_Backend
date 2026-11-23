import hashlib
import unittest
from datetime import timedelta
from unittest.mock import patch

import test_admin_service as fixtures
import test_station_service as station_fixtures
from conversation_history import backfill_portal_history
from farmer_service import WeatherSettings, iso, utcnow
from meta_whatsapp import phone_aliases, phone_id
from whatsapp_worker import WhatsAppWorker


class HistoryFarmTests(unittest.TestCase):
    setUp = fixtures.AdminTests.setUp
    farmer = fixtures.AdminTests.farmer
    admin = fixtures.AdminTests.admin
    login = fixtures.AdminTests.login
    headers = fixtures.AdminTests.headers
    station = station_fixtures.StationTests.station
    staff = station_fixtures.StationTests.staff

    def account(self, phone="254700000001"):
        user = self.farmer(phone)
        token = self.module.create_token({"phone": phone})
        return user, {"Authorization": "Bearer " + token}

    def farm(self, user, farm_id="farm"):
        return self.containers["farms"].create_item({
            "id": farm_id, "owner": user["phoneNumber"], "name": farm_id,
            "crop": "Maize", "acreage": 1, "latitude": 0, "longitude": 0,
        })

    def test_phone_aliases_resolve_one_account_and_legacy_local_farms(self):
        user, _ = self.account()
        farm = self.farm(user)
        farm["owner"] = "0700000001"
        self.containers["farms"].upsert_item(farm)
        for value in ("+254700000001", "254700000001", "0700000001", "+254 (700) 000-001"):
            self.assertEqual(phone_id(value), "254700000001")
            self.assertEqual(self.farmers.find_user(value)["id"], user["id"])
            headers = {"Authorization": "Bearer " + self.module.create_token({"phone": value})}
            response = self.client.get("/farms", headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()[0]["id"], "farm")
        self.assertIn("0700000001", phone_aliases("+254700000001"))
        self.assertEqual(phone_id("0112345678"), "254112345678")
        self.assertEqual(phone_id("+447123456789"), "447123456789")
        with self.assertRaises(ValueError):
            phone_id("07123")
        self.containers["users"].create_item({"id": "0700000001", "phoneNumber": "0700000001"})
        with self.assertRaisesRegex(Exception, "409"):
            self.farmers.find_user("254700000001")

    def test_farm_archiving_restore_ownership_revisions_and_primary_guard(self):
        user, headers = self.account()
        farm = self.farm(user)
        other = self.farm(user, "other")
        path = "/profiles/254700000001/farms/farm/status"
        self.assertEqual(self.client.put(path, headers=headers, json={
            "active": False, "revision": farm["_etag"],
        }).status_code, 409)
        self.client.put("/profiles/254700000001/primary-farm", headers=headers, json={"farmId": "other"})
        response = self.client.put(path, headers=headers, json={"active": False, "revision": farm["_etag"]})
        self.assertEqual(response.status_code, 200, response.text)
        archived = response.json()
        snapshot = self.client.get("/farmer-services", headers=headers).json()
        self.assertEqual([f["id"] for f in snapshot["farms"]], ["other"])
        self.assertEqual([f["id"] for f in snapshot["archivedFarms"]], ["farm"])
        self.assertEqual(snapshot["primaryFarmId"], other["id"])
        self.assertEqual(self.farmers.active_farm(self.farmers.find_user(user["phoneNumber"]))["id"], "other")
        self.assertEqual(self.client.put(path, headers=headers, json={
            "active": True, "revision": farm["_etag"],
        }).status_code, 409)
        _, foreign_headers = self.account("254700000002")
        self.assertEqual(self.client.put(path, headers=foreign_headers, json={
            "active": True, "revision": archived["revision"],
        }).status_code, 403)
        restored = self.client.put(path, headers=headers, json={"active": True, "revision": archived["revision"]})
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertNotIn("archivedAt", restored.json())
        self.assertEqual(len(self.client.get("/farms", headers=headers).json()), 2)

    def test_archived_last_farm_pauses_forecasts_and_retains_history(self):
        user, headers = self.account()
        farm = self.farm(user)
        self.farmers.update_weather(user, WeatherSettings(enabled=True, consent=True, farmId="farm"), user["phoneNumber"])
        user["weather"]["nextDueAt"] = iso(utcnow() - timedelta(minutes=1))
        self.containers["users"].upsert_item(user)
        self.farmers.save_chat(user, farm, "user", "Keep this history", "old-message")
        response = self.client.put("/profiles/254700000001/farms/farm/status", headers=headers, json={
            "active": False, "revision": farm["_etag"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        snapshot = self.client.get("/farmer-services", headers=headers).json()
        self.assertEqual(snapshot["farms"], [])
        self.assertIsNone(snapshot["primaryFarmId"])
        self.assertIsNone(snapshot["station"])
        self.assertEqual(len(self.containers["chats"].items), 1)
        with patch.object(self.farmers, "forecast") as forecast:
            result = WhatsAppWorker(self.farmers).deliver({
                "owner": user["phoneNumber"], "kind": "weather", "revision": user["weather"]["revision"],
                "createdAt": iso(utcnow()),
            })
        self.assertIsNone(result)
        forecast.assert_not_called()
        self.assertFalse(self.module.station_service.available("drones", response.json()))
        worker = WhatsAppWorker(self.farmers)
        worker.schedule()
        self.assertEqual(len(self.containers["whatsapp_jobs"].items), 0)
        restored = self.client.put("/profiles/254700000001/farms/farm/status", headers=headers, json={
            "active": True, "revision": response.json()["revision"],
        })
        self.assertEqual(restored.status_code, 200, restored.text)
        worker.schedule()
        self.assertEqual(len(self.containers["whatsapp_jobs"].items), 1)

    def test_latest_page_and_older_pages_keep_all_equal_timestamp_messages(self):
        user, headers = self.account()
        timestamp = iso(utcnow() - timedelta(minutes=1))
        for index in range(63):
            self.containers["whatsapp_interactions"].create_item({
                "id": str(index), "owner": user["phoneNumber"], "direction": "inbound",
                "message": f"Message {index}", "timestamp": timestamp, "status": "received",
            })
        ids, cursor = [], None
        while True:
            response = self.client.get("/chat-history/page", headers=headers,
                                       params={"limit": 7, **({"cursor": cursor} if cursor else {})})
            self.assertEqual(response.status_code, 200, response.text)
            page = response.json()
            ids.extend(m["id"] for m in page["items"])
            cursor = page["cursor"]
            if not cursor:
                break
        self.assertEqual(len(ids), 63)
        self.assertEqual(len(set(ids)), 63)
        self.containers["whatsapp_interactions"].create_item({
            "id": "latest", "owner": user["phoneNumber"], "direction": "outbound",
            "message": "Latest reply", "timestamp": iso(utcnow()), "status": "accepted",
        })
        with patch.object(self.containers["whatsapp_interactions"], "query_items",
                          wraps=self.containers["whatsapp_interactions"].query_items) as query:
            page = self.client.get("/chat-history/page?limit=1", headers=headers).json()
        self.assertEqual(query.call_args.kwargs["partition_key"], user["phoneNumber"])
        self.assertEqual(page["items"][-1]["id"], "latest")

    def test_history_excludes_internal_drafts_and_other_farmers_and_binds_cursor(self):
        user, headers = self.account()
        _, other_headers = self.account("254700000002")
        for index, (owner, direction, status) in enumerate([
            (user["phoneNumber"], "inbound", "received"), (user["phoneNumber"], "outbound", "accepted"),
            (user["phoneNumber"], "internal", "recorded"), (user["phoneNumber"], "outbound", "cancelled"),
            (user["phoneNumber"], "outbound", "sending"), ("254700000002", "inbound", "received"),
        ]):
            self.containers["whatsapp_interactions"].create_item({
                "id": str(index), "owner": owner, "direction": direction,
                "status": status, "message": "Message", "timestamp": iso(utcnow()),
            })
        first = self.client.get("/chat-history/page?limit=1", headers=headers).json()
        self.assertIsNotNone(first["cursor"])
        self.assertEqual(self.client.get("/chat-history/page", headers=other_headers,
                                        params={"cursor": first["cursor"]}).status_code, 409)
        self.assertEqual(self.client.get("/chat-history/page", headers=headers,
                                        params={"cursor": "invalid"}).status_code, 400)
        all_items = self.client.get("/chat-history/page", headers=headers).json()["items"]
        self.assertEqual({i["id"] for i in all_items}, {"0", "1"})

    def test_legacy_import_is_idempotent_and_does_not_duplicate_whatsapp_advice(self):
        user, headers = self.account()
        self.farm(user)
        self.farmers.record_inbound({"id": "meta-inbound", "from": user["phoneNumber"],
                                     "type": "text", "text": {"body": "Hello"}})
        for role in ("user", "assistant"):
            self.containers["chats"].create_item({
                "id": hashlib.sha256(f"meta-inbound:{role}".encode()).hexdigest(),
                "user": user["phoneNumber"], "role": role, "message": "Advice copy", "timestamp": iso(utcnow()),
            })
        self.containers["chats"].create_item({
            "id": "legacy", "user": "0700000001", "role": "assistant", "message": "Legacy advice",
            "timestamp": "2025-10-12T18:39:48",
        })
        self.containers["chats"].create_item({
            "id": "structured", "user": {"phone": "+254700000001", "name": "Private metadata", "exp": 1},
            "role": "assistant", "message": "Structured owner advice", "timestamp": "2025-11-12T18:39:48",
        })
        self.containers["chats"].create_item({
            "id": "unowned", "user": {}, "role": "assistant", "message": "Unowned history",
            "timestamp": "2025-11-12T18:39:48",
        })
        self.containers["whatsapp_interactions"].create_item({
            "id": "admin-audit", "owner": "station-config", "direction": "internal", "timestamp": iso(utcnow()),
        })
        with self.assertLogs("conversation_history", level="WARNING"):
            result = backfill_portal_history(self.containers["chats"], self.containers["whatsapp_interactions"])
        self.assertEqual(result, {"imported": 2, "alreadyImported": 0, "whatsappAdviceCopiesSkipped": 2, "invalidOwnersSkipped": 1})
        with self.assertLogs("conversation_history", level="WARNING"):
            again = backfill_portal_history(self.containers["chats"], self.containers["whatsapp_interactions"])
        self.assertEqual(again["alreadyImported"], 2)
        messages = self.client.get("/chat-history/page", headers=headers).json()["items"]
        self.assertEqual([m["message"] for m in messages], ["Legacy advice", "Structured owner advice", "Hello"])
        self.assertTrue(messages[0]["timestamp"].endswith("+00:00"))
        self.assertNotIn("Private metadata", str(messages))
        self.assertEqual(len(self.containers["chats"].items), 5)

    def test_future_portal_messages_are_saved_to_both_histories(self):
        user, headers = self.account()
        response = self.client.post("/chat", headers=headers, json={"message": "Portal question"})
        self.assertEqual(response.status_code, 200, response.text)
        page = self.client.get("/chat-history/page", headers=headers).json()
        self.assertEqual(page["items"][0]["message"], "Portal question")
        self.assertEqual(page["items"][0]["channel"], "portal")
        self.assertEqual(len(self.containers["chats"].items), 1)

    def test_geographic_history_scope_changes_and_station_archive_authorization(self):
        admin = self.admin()
        station = self.station(admin)
        user, headers = self.account()
        self.farm(user, "north")
        south = self.farm(user, "south")
        self.containers["farms"].upsert_item({**south, "latitude": 1})
        for farm_id in ("north", "south"):
            for index in range(2):
                self.containers["whatsapp_interactions"].create_item({
                    "id": f"{farm_id}-{index}", "owner": user["phoneNumber"], "direction": "inbound",
                    "farmId": farm_id, "message": "Farm question", "timestamp": iso(utcnow()), "status": "received",
                })
        rule = self.client.put("/admin/services/chat", headers=admin, json={
            "enabled": True, "scope": "stations", "stationIds": [station["id"]], "revision": "",
        }).json()
        page = self.client.get("/chat-history/page", headers=headers).json()
        self.assertEqual({i["id"] for i in page["items"]}, {"north-0", "north-1"})
        first = self.client.get("/chat-history/page?limit=1", headers=headers).json()
        self.client.put("/admin/services/chat", headers=admin, json={
            "enabled": True, "scope": "all", "stationIds": [], "revision": rule["revision"],
        })
        self.assertEqual(self.client.get("/chat-history/page", headers=headers,
                                        params={"cursor": first["cursor"]}).status_code, 409)
        staff, _ = self.staff(admin, station["id"])
        outside = self.farmer("254700000002")
        farm = self.farm(outside, "outside")
        farm = self.containers["farms"].upsert_item({**farm, "latitude": 1})
        self.assertEqual(self.client.put("/profiles/254700000002/farms/outside/status", headers=staff,
                                        json={"active": False, "revision": farm["_etag"]}).status_code, 403)

    def test_history_does_not_expose_portal_login_codes(self):
        user, headers = self.account()
        self.farmers.send_recorded(user["phoneNumber"], "Your portal code: 123456", lambda: "meta-code")
        items = self.client.get("/chat-history/page", headers=headers).json()["items"]
        self.assertEqual(items[0]["message"], "Your portal code: [redacted]")

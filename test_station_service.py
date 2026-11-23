import math
import unittest
from unittest.mock import patch

import test_admin_service as fixtures
from farmer_service import WeatherSettings, iso, utcnow
from station_service import distance_km, farm_point
from whatsapp_worker import WhatsAppWorker


class StationTests(unittest.TestCase):
    setUp = fixtures.AdminTests.setUp
    login = fixtures.AdminTests.login
    headers = fixtures.AdminTests.headers
    admin = fixtures.AdminTests.admin
    farmer = fixtures.AdminTests.farmer

    def station(self, headers, name="North", latitude=0, longitude=0, radius=20):
        response = self.client.post("/admin/stations", headers=headers, json={
            "name": name, "latitude": latitude, "longitude": longitude, "radiusKm": radius,
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def farm(self, user, farm_id, latitude=0, longitude=0):
        return self.containers["farms"].create_item({
            "id": farm_id, "owner": user["phoneNumber"], "name": farm_id,
            "crop": "Maize", "acreage": 2, "latitude": latitude, "longitude": longitude,
        })

    def farmer_headers(self, user):
        return {"Authorization": "Bearer " + self.module.create_token({
            "phone": user["phoneNumber"], "name": user["name"],
        })}

    def staff(self, admin, station_id):
        response = self.client.post("/admin/staff", headers=admin, json={
            "email": "station@example.invalid", "name": "Station officer",
            "password": self.initial, "stationId": station_id,
        })
        self.assertEqual(response.status_code, 201, response.text)
        login = self.login("station@example.invalid")
        changed = self.client.post("/admin/password", headers=self.headers(login), json={
            "currentPassword": self.initial, "newPassword": self.changed,
        })
        self.assertEqual(changed.status_code, 200, changed.text)
        return self.headers(changed), response.json()

    def test_radius_nearest_boundary_and_unlocated_farms(self):
        admin = self.admin()
        north = self.station(admin)
        east = self.station(admin, "East", longitude=0.1)
        access = self.module.station_service
        self.assertEqual(access.covering({"latitude": 0, "longitude": 0.09}, access.stations())[0]["id"], east["id"])
        self.assertEqual(access.covering({"latitude": 0, "longitude": 0}, access.stations())[0]["id"], north["id"])
        boundary = {"latitude": 0, "longitude": math.degrees(20 / 6371.0088)}
        self.assertAlmostEqual(distance_km((0, boundary["longitude"]), north), 20)
        self.assertIn(north["id"], [s["id"] for s in access.covering(boundary, access.stations())])
        self.assertEqual(access.covering({"latitude": 0, "longitude": 1}, [north]), [])
        self.assertEqual(access.covering({}, access.stations()), [])
        self.assertIsNone(farm_point({"coordinates": [{"invalid": True}]}))
        self.assertEqual(farm_point({"coordinates": [{"lat": 0, "lng": 0.1}]}), (0, 0.1))

    def test_primary_farm_and_override_do_not_expand_physical_eligibility(self):
        admin = self.admin()
        north = self.station(admin)
        south = self.station(admin, "South", latitude=1)
        user = self.farmer()
        self.farm(user, "a-north")
        self.farm(user, "b-south", latitude=1)
        farmer = self.farmer_headers(user)
        snapshot = self.client.get("/farmer-services", headers=farmer).json()
        self.assertEqual(snapshot["station"]["id"], north["id"])
        changed = self.client.put(f"/profiles/{user['phoneNumber']}/primary-farm", headers=farmer,
                                  json={"farmId": "b-south"})
        self.assertEqual(changed.json()["station"]["id"], south["id"])
        self.assertEqual(self.client.put(
            f"/profiles/{user['phoneNumber']}/primary-farm", headers=farmer,
            json={"farmId": "someone-elses-farm"},
        ).status_code, 404)
        saved = self.farmers.find_user(user["phoneNumber"])
        response = self.client.patch(f"/admin/farmers/{user['phoneNumber']}", headers=admin, json={
            "name": user["name"], "language": "en", "active": True, "revision": saved["_etag"],
            "stationOverride": north["id"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        rule = self.client.put("/admin/services/drones", headers=admin, json={
            "enabled": True, "scope": "stations", "stationIds": [north["id"]], "revision": "",
        })
        self.assertEqual(rule.status_code, 200, rule.text)
        snapshot = self.client.get("/farmer-services", headers=farmer).json()
        self.assertEqual(snapshot["station"]["id"], north["id"])
        self.assertIn("drones", snapshot["farms"][0]["services"])
        self.assertNotIn("drones", snapshot["farms"][1]["services"])

    def test_station_scope_filters_before_pagination_and_protects_direct_access(self):
        admin = self.admin()
        north = self.station(admin)
        outside = self.farmer("254700000002")
        inside = self.farmer()
        self.farm(outside, "outside", latitude=1)
        self.farm(inside, "inside")
        for owner in (outside["phoneNumber"], inside["phoneNumber"]):
            self.containers["whatsapp_interactions"].create_item({
                "id": owner, "owner": owner, "timestamp": iso(utcnow()), "mediaId": "media",
                "direction": "inbound", "type": "image", "status": "received",
            })
            self.containers["chats"].create_item({
                "id": owner, "user": owner, "mediaId": "media", "timestamp": iso(utcnow()),
            })
        staff, _ = self.staff(admin, north["id"])
        listed = self.client.get("/admin/farmers?limit=1", headers=staff)
        self.assertEqual(listed.status_code, 200, listed.text)
        self.assertEqual([u["phoneNumber"] for u in listed.json()["items"]], [inside["phoneNumber"]])
        self.assertEqual(listed.json()["items"][0]["stationName"], north["name"])
        inbox = self.client.get("/staff/interactions?limit=1", headers=staff).json()
        self.assertEqual([i["owner"] for i in inbox["items"]], [inside["phoneNumber"]])
        self.assertIsNone(inbox["cursor"])
        data = self.client.get("/admin/dashboard", headers=staff).json()
        self.assertEqual(data["farmers"], 1)
        self.assertEqual(data["farms"], 1)
        self.assertTrue(all(i["owner"] == inside["phoneNumber"] for i in data["recent"]))
        for path in (
            f"/profiles/{outside['phoneNumber']}", f"/profiles/{outside['phoneNumber']}/farms",
            f"/staff/conversations/{outside['phoneNumber']}",
            f"/staff/interactions/{outside['phoneNumber']}/image",
            f"/staff/interactions/{outside['phoneNumber']}/image?source=advice",
            f"/staff/interactions?phone={outside['phoneNumber']}",
        ):
            response = self.client.get(path, headers=staff)
            self.assertEqual(response.status_code, 403, (path, response.text))
        self.assertEqual(self.client.patch(f"/profiles/{outside['phoneNumber']}", headers=staff,
                                          json={"name": "Changed", "language": "en"}).status_code, 403)
        self.assertEqual(self.client.patch(f"/profiles/{inside['phoneNumber']}", headers=staff,
                                          json={"name": "Updated", "language": "en"}).status_code, 200)
        self.assertEqual(self.client.put(f"/staff/conversations/{outside['phoneNumber']}", headers=staff,
                                        json={"paused": True, "revision": ""}).status_code, 403)
        self.assertEqual(self.client.post(f"/staff/conversations/{outside['phoneNumber']}/messages", headers=staff,
                                         json={"message": "Test", "revision": "",
                                               "requestId": "12345678-1234-4234-8234-123456789012"}).status_code, 403)
        self.farmers.messenger.download_image.assert_not_called()
        self.assertEqual(self.client.get("/farms", headers=staff).status_code, 403)
        self.assertEqual(self.client.get("/admin/services", headers=staff).status_code, 403)
        self.assertEqual(len(self.client.get("/admin/farmers", headers=admin).json()["items"]), 2)

    def test_archiving_station_immediately_removes_access_and_coverage(self):
        admin = self.admin()
        station = self.station(admin)
        user = self.farmer()
        self.farm(user, "farm")
        staff, _ = self.staff(admin, station["id"])
        rules = self.client.put("/admin/services/drones", headers=admin, json={
            "enabled": True, "scope": "stations", "stationIds": [station["id"]], "revision": "",
        }).json()
        self.assertEqual(self.client.get(f"/profiles/{user['phoneNumber']}", headers=staff).status_code, 200)
        updated = self.client.patch("/admin/stations/" + station["id"], headers=admin, json={
            key: value for key, value in {**station, "active": False}.items() if key not in {"id", "createdAt"}
        })
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(self.client.get(f"/profiles/{user['phoneNumber']}", headers=staff).status_code, 403)
        self.assertEqual(self.client.get("/admin/dashboard", headers=staff).status_code, 403)
        self.assertEqual(self.client.get("/admin/session", headers=staff).status_code, 200)
        snapshot = self.client.get("/farmer-services", headers=self.farmer_headers(user)).json()
        self.assertIsNone(snapshot["station"])
        self.assertEqual(self.client.put("/admin/services/drones", headers=admin, json={
            "enabled": False, "scope": "stations", "stationIds": [station["id"]], "revision": rules["revision"],
        }).status_code, 200)

    def test_reassignment_revokes_session_and_primary_change_is_not_allowed_to_staff(self):
        admin = self.admin()
        north = self.station(admin)
        south = self.station(admin, "South", latitude=1)
        user = self.farmer()
        self.farm(user, "farm")
        staff, account = self.staff(admin, north["id"])
        self.assertEqual(self.client.put(f"/profiles/{user['phoneNumber']}/primary-farm", headers=staff,
                                        json={"farmId": "farm"}).status_code, 403)
        account = self.module.admin_service.find(account["id"])
        moved = self.client.patch("/admin/staff/" + account["id"], headers=admin, json={
            "name": account["name"], "role": "staff", "active": True,
            "stationId": south["id"], "revision": account["_etag"],
        })
        self.assertEqual(moved.status_code, 200, moved.text)
        self.assertEqual(self.client.get("/admin/session", headers=staff).status_code, 401)
        staff = self.headers(self.login("station@example.invalid", self.changed))
        self.assertEqual(self.client.get(f"/profiles/{user['phoneNumber']}", headers=staff).status_code, 403)

    def test_service_settings_persist_validate_revisions_and_gate_old_endpoints(self):
        admin = self.admin()
        user = self.farmer()
        self.farm(user, "farm")
        farmer = self.farmer_headers(user)
        initial = self.client.get("/farmer-services", headers=farmer).json()
        self.assertEqual(set(initial["services"]), {"weather", "chat", "reports"})
        self.assertEqual(self.client.get("/grid-images", headers=farmer).status_code, 403)
        self.assertEqual(self.client.post("/soil-samples", headers=farmer, json={}).status_code, 403)
        disabled = self.client.put("/admin/services/reports", headers=admin, json={
            "enabled": False, "scope": "all", "stationIds": [], "revision": "",
        })
        self.assertEqual(disabled.status_code, 200, disabled.text)
        self.assertEqual(self.client.get("/harvests", headers=farmer).status_code, 403)
        self.assertNotIn("reports", self.client.get("/farmer-services", headers=farmer).json()["services"])
        self.assertEqual(self.client.put("/admin/services/chat", headers=admin, json={
            "enabled": False, "scope": "all", "stationIds": [], "revision": "",
        }).status_code, 409)
        revision = disabled.json()["revision"]
        response = self.client.put("/admin/services/weather", headers=admin, json={
            "enabled": False, "scope": "all", "stationIds": [], "revision": revision,
        })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.client.put(f"/profiles/{user['phoneNumber']}/weather", headers=farmer, json={
            "enabled": True, "consent": True, "farmId": "farm",
        }).status_code, 403)
        self.assertEqual(self.client.get("/admin/services", headers=admin).json()["items"][0]["enabled"], False)
        self.assertEqual(self.client.put("/admin/services/drones", headers=admin, json={
            "enabled": True, "scope": "all", "stationIds": [], "revision": response.json()["revision"],
        }).status_code, 422)

    def test_station_validation_and_stale_edits(self):
        admin = self.admin()
        station = self.station(admin)
        body = {key: value for key, value in station.items() if key not in {"id", "createdAt"}}
        changed = self.client.patch("/admin/stations/" + station["id"], headers=admin,
                                    json={**body, "radiusKm": 30})
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(self.client.patch("/admin/stations/" + station["id"], headers=admin,
                                          json=body).status_code, 409)
        self.assertEqual(self.client.post("/admin/stations", headers=admin, json={
            "name": "Invalid", "latitude": 91, "longitude": 0, "radiusKm": 0,
        }).status_code, 422)
        with patch.dict("os.environ", {"STAFF_PHONES": "254700000001"}):
            user = self.farmer()
            other = self.farmer("254700000002")
            self.assertEqual(self.client.get(
                f"/profiles/{other['phoneNumber']}", headers=self.farmer_headers(user),
            ).status_code, 403)

    def test_disabling_weather_cancels_an_already_queued_forecast(self):
        admin = self.admin()
        user = self.farmer()
        self.farm(user, "farm")
        self.farmers.update_weather(
            user, WeatherSettings(enabled=True, consent=True, farmId="farm"), {"phone": user["phoneNumber"]},
        )
        response = self.client.put("/admin/services/weather", headers=admin, json={
            "enabled": False, "scope": "all", "stationIds": [], "revision": "",
        })
        self.assertEqual(response.status_code, 200, response.text)
        with patch.object(self.farmers, "forecast") as forecast:
            result = WhatsAppWorker(self.farmers).deliver({
                "owner": user["phoneNumber"], "kind": "weather", "revision": user["weather"]["revision"],
                "createdAt": iso(utcnow()),
            })
        self.assertIsNone(result)
        forecast.assert_not_called()
        self.farmers.messenger.template.assert_not_called()

    def test_geographic_digital_services_only_return_eligible_farm_records(self):
        admin = self.admin()
        station = self.station(admin)
        user = self.farmer()
        for farm_id, latitude in (("north", 0), ("south", 1)):
            self.farm(user, farm_id, latitude=latitude)
            self.containers["chats"].create_item({
                "id": farm_id, "user": user["phoneNumber"], "farmId": farm_id, "timestamp": iso(utcnow()),
            })
            self.containers["harvests"].create_item({
                "id": farm_id, "owner": user["phoneNumber"], "farmId": farm_id,
            })
        revision = ""
        for key in ("chat", "reports"):
            result = self.client.put("/admin/services/" + key, headers=admin, json={
                "enabled": True, "scope": "stations", "stationIds": [station["id"]], "revision": revision,
            })
            self.assertEqual(result.status_code, 200, result.text)
            revision = result.json()["revision"]
        farmer = self.farmer_headers(user)
        self.assertEqual([r["farmId"] for r in self.client.get("/chat-history", headers=farmer).json()], ["north"])
        self.assertEqual([r["farmId"] for r in self.client.get("/harvests", headers=farmer).json()], ["north"])
        self.assertEqual(self.client.post("/harvests", headers=farmer, json={
            "farmId": "south", "date": "2026-11-20", "yield": 5, "crop": "Maize",
        }).status_code, 403)

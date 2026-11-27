import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from admin_service import account_id, check_password, password_hash
from farmer_service import AdviceUnavailableError, iso, utcnow
from google.api_core.exceptions import NotFound
from test_farmer_service import Container
from whatsapp_worker import WhatsAppWorker


class AdminTests(unittest.TestCase):
    def setUp(self):
        self.initial = "Test-temporary-password-1"
        self.changed = "Test-replacement-password-2"
        self.email = "admin@example.invalid"
        self.containers = {
            name: Container() for name in (
                "users", "farms", "chats", "whatsapp_jobs", "whatsapp_interactions",
            )
        }
        cosmos = Mock()
        cosmos.get_database_client.return_value.get_container_client.side_effect = (
            lambda name: self.containers.setdefault(name, Container())
        )
        self.settings = patch.dict(os.environ, {
            "AZURE_ENVIRONMENT": "production", "GOOGLE_API_KEY": "test-key",
            "TOKEN_ALGORITHM": "HS256", "TOKEN_SECRET_KEY": "test-secret",
            "COSMOS_ENDPOINT": "https://example.invalid", "COSMOS_KEY": "test-key",
            "DATABASE_NAME": "test-db", "AZURE_STORAGE_CONNECTION_STRING": "test-storage",
            "ADMIN_BOOTSTRAP_EMAIL": self.email,
            "ADMIN_BOOTSTRAP_PASSWORD_HASH": password_hash(self.initial),
            "STAFF_PHONES": "",
        })
        self.settings.start()
        self.addCleanup(self.settings.stop)
        with patch("azure.cosmos.CosmosClient", return_value=cosmos), patch(
            "azure.storage.blob.BlobServiceClient.from_connection_string"
        ):
            spec = importlib.util.spec_from_file_location(
                "admin_test_app", Path(__file__).with_name("main.py"),
            )
            self.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.module)
        self.client = TestClient(self.module.app)
        self.users = self.containers["users"]
        self.farmers = self.module.farmer_service
        self.farmers.messenger = Mock()
        self.farmers.adviser = Mock()

    def login(self, email=None, password=None):
        return self.client.post("/admin/login", json={
            "email": email or self.email, "password": password or self.initial,
        })

    def headers(self, result):
        return {"Authorization": f"Bearer {result.json()['token']}"}

    def admin(self):
        initial = self.login()
        self.assertEqual(initial.status_code, 200, initial.text)
        changed = self.client.post("/admin/password", headers=self.headers(initial), json={
            "currentPassword": self.initial, "newPassword": self.changed,
        })
        self.assertEqual(changed.status_code, 200, changed.text)
        return self.headers(changed)

    def create_staff(self, headers, role="staff"):
        response = self.client.post("/admin/staff", headers=headers, json={
            "email": "support@example.invalid", "name": "Support", "role": role,
            "password": self.initial,
        })
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def farmer(self, phone="254700000001"):
        return self.users.create_item({
            "id": phone, "phoneNumber": phone, "name": "Farmer", "verified": True,
            "onboardingStep": "ready", "language": "en",
        })

    def test_temporary_password_gate_hashing_and_bootstrap_never_overwrites(self):
        response = self.login(email="ADMIN@EXAMPLE.INVALID")
        self.assertEqual(response.status_code, 200, response.text)
        headers = self.headers(response)
        self.assertTrue(response.json()["account"]["passwordChangeRequired"])
        self.assertNotIn("passwordHash", response.text)
        self.assertEqual(self.client.get("/admin/dashboard", headers=headers).status_code, 403)
        self.assertEqual(self.client.get("/staff/interactions", headers=headers).status_code, 403)
        self.assertEqual(self.client.get("/admin/session", headers=headers).status_code, 200)
        original = self.module.admin_service.find(account_id(self.email))["passwordHash"]
        self.assertNotEqual(original, self.initial)
        self.assertTrue(check_password(self.initial, original))
        with patch.dict(os.environ, {"ADMIN_BOOTSTRAP_PASSWORD_HASH": password_hash(self.changed)}):
            self.assertEqual(self.login(password=self.changed).status_code, 401)
        self.assertEqual(self.login(password="incorrect").json()["detail"], "Invalid email or password.")
        self.assertEqual(self.login(email="nobody@example.invalid").json()["detail"], "Invalid email or password.")

    def test_ai_provider_failure_and_empty_response_are_explicit(self):
        for error in (NotFound("model unavailable"), None):
            with self.subTest(error=type(error).__name__):
                adviser = Mock()
                adviser.generate_content.side_effect = error
                adviser.generate_content.return_value.text = ""
                with patch.object(self.module.genai, "GenerativeModel", return_value=adviser):
                    with self.assertRaises(AdviceUnavailableError):
                        self.module.advise_farmer(
                            {"language": "en"}, {"id": "farm"}, [], "Inspect this crop.",
                            (b"synthetic-image", "image/png"),
                        )

    def test_role_boundaries_and_password_change_revokes_old_token(self):
        admin = self.admin()
        self.create_staff(admin)
        login = self.login(email="support@example.invalid")
        old_headers = self.headers(login)
        changed = self.client.post("/admin/password", headers=old_headers, json={
            "currentPassword": self.initial, "newPassword": self.changed,
        })
        staff = self.headers(changed)
        self.assertEqual(self.client.get("/admin/session", headers=old_headers).status_code, 401)
        self.assertEqual(self.client.get("/admin/dashboard", headers=staff).status_code, 403)
        self.assertEqual(self.client.get("/staff/interactions", headers=staff).status_code, 403)
        for path in ("/admin/staff", "/admin/farmers"):
            self.assertEqual(self.client.get(path, headers=staff).status_code, 403)
            self.assertEqual(self.client.get(path).status_code, 401)
        farmer = self.farmer()
        token = self.module.create_token({"phone": farmer["phoneNumber"], "name": farmer["name"]})
        self.assertEqual(self.client.get(
            "/admin/dashboard", headers={"Authorization": f"Bearer {token}"},
        ).status_code, 403)

    def test_staff_crud_duplicate_email_etags_archiving_and_restore(self):
        admin = self.admin()
        created = self.create_staff(admin)
        duplicate = self.client.post("/admin/staff", headers=admin, json={
            "email": "SUPPORT@EXAMPLE.INVALID", "name": "Duplicate", "password": self.initial,
        })
        self.assertEqual(duplicate.status_code, 409)
        page = self.client.get("/admin/staff?limit=1", headers=admin).json()
        self.assertEqual(len(page["items"]), 1)
        self.assertIsNotNone(page["cursor"])
        login = self.login(email=created["email"])
        stale = created["revision"]
        values = {key: created[key] for key in ("name", "role", "active", "revision")}
        current = self.client.get("/admin/staff", headers=admin).json()["items"]
        values["revision"] = next(item for item in current if item["id"] == created["id"])["revision"]
        updated = self.client.patch(f"/admin/staff/{created['id']}", headers=admin, json=values)
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(self.client.get("/admin/session", headers=self.headers(login)).status_code, 401)
        self.assertEqual(self.client.delete(
            f"/admin/staff/{created['id']}", headers=admin, params={"revision": stale},
        ).status_code, 409)
        deleted = self.client.delete(
            f"/admin/staff/{created['id']}", headers=admin,
            params={"revision": updated.json()["revision"]},
        )
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertFalse(deleted.json()["active"])
        self.assertEqual(self.login(email=created["email"]).status_code, 401)
        values.update(active=True, revision=deleted.json()["revision"])
        restored = self.client.patch(f"/admin/staff/{created['id']}", headers=admin, json=values)
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(self.login(email=created["email"]).status_code, 200)
        self.assertNotIn(self.initial, str(self.containers["whatsapp_interactions"].items))

    def test_initial_administrator_cannot_be_archived_or_demoted(self):
        headers = self.admin()
        account = self.client.get("/admin/session", headers=headers).json()
        path = f"/admin/staff/{account['id']}"
        self.assertEqual(self.client.delete(
            path, headers=headers, params={"revision": account["revision"]},
        ).status_code, 409)
        values = {
            "name": account["name"], "role": "staff", "active": True, "revision": account["revision"],
        }
        self.assertEqual(self.client.patch(path, headers=headers, json=values).status_code, 409)
        values["role"] = "admin"
        values["name"] = "Renamed administrator"
        self.assertEqual(self.client.patch(path, headers=headers, json=values).status_code, 200)
        self.assertEqual(self.client.get("/admin/dashboard", headers=headers).status_code, 200)

    def test_farmer_crud_preserves_consent_and_revokes_access_and_automation(self):
        headers = self.admin()
        values = {"phoneNumber": "+254700000001", "name": "Farmer", "language": "sw"}
        response = self.client.post("/admin/farmers", headers=headers, json=values)
        self.assertEqual(response.status_code, 201, response.text)
        farmer = response.json()
        self.assertFalse(farmer["verified"])
        self.assertIsNone(farmer["consentAt"])
        self.assertEqual(farmer["onboardingStep"], "consent")
        self.assertEqual(self.client.post("/admin/farmers", headers=headers, json=values).status_code, 409)
        token = self.module.create_token({"phone": farmer["phoneNumber"], "name": farmer["name"]})
        farmer_headers = {"Authorization": f"Bearer {token}"}
        self.containers["farms"].create_item({"id": "farm1", "owner": farmer["phoneNumber"]})
        deleted = self.client.delete(
            f"/admin/farmers/{farmer['phoneNumber']}", headers=headers,
            params={"revision": farmer["revision"]},
        )
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(self.client.get("/me", headers=farmer_headers).status_code, 401)
        self.assertEqual(self.client.post("/login", json={"phone": farmer["phoneNumber"]}).status_code, 403)
        job = {
            "id": "queued", "owner": farmer["phoneNumber"], "kind": "inbound",
            "message": {"id": "queued", "from": farmer["phoneNumber"], "type": "text",
                        "text": {"body": "Hello"}}, "createdAt": iso(utcnow()),
        }
        self.assertIsNone(WhatsAppWorker(self.farmers).prepare_inbound(job))
        self.farmers.adviser.assert_not_called()
        self.farmers.messenger.text.assert_not_called()
        self.assertIn("farm1", self.containers["farms"].items)
        restored = self.client.patch(
            f"/admin/farmers/{farmer['phoneNumber']}", headers=headers, json={
                "name": "Updated", "language": "en", "active": True,
                "revision": deleted.json()["revision"],
            },
        )
        self.assertEqual(restored.status_code, 200, restored.text)
        self.assertEqual(self.client.get("/me", headers=farmer_headers).status_code, 401)
        self.assertTrue(self.farmers.find_user(farmer["phoneNumber"])["handoff"]["paused"])
        listed = self.client.get("/admin/farmers", headers=headers).json()["items"]
        self.assertEqual([u["name"] for u in listed], ["Updated"])
        self.assertNotIn("codeHash", str(listed))

    def test_invalid_inputs_cannot_set_roles_consent_or_plaintext_password(self):
        headers = self.admin()
        for values in (
            {"phoneNumber": "not-a-phone", "name": "Farmer"},
            {"phoneNumber": "254700000001", "name": "Farmer", "verified": True},
            {"phoneNumber": "254700000001", "name": "Farmer", "consentAt": iso(utcnow())},
        ):
            self.assertEqual(self.client.post("/admin/farmers", headers=headers, json=values).status_code, 422)
        self.assertEqual(self.client.post("/admin/staff", headers=headers, json={
            "email": "support@example.invalid", "name": "Support", "password": "short",
        }).status_code, 422)
        self.assertEqual(self.client.post("/admin/password", headers=headers, json={
            "currentPassword": "wrong", "newPassword": self.initial,
        }).status_code, 401)

    def test_staff_password_is_not_silently_trimmed(self):
        headers = self.admin()
        password = "  A temporary password with spaces  "
        created = self.client.post("/admin/staff", headers=headers, json={
            "email": "spaces@example.invalid", "name": "Spaces", "password": password,
        })
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(self.login(email="spaces@example.invalid", password=password).status_code, 200)
        self.assertEqual(self.login(email="spaces@example.invalid", password=password.strip()).status_code, 401)
        self.assertEqual(self.client.post("/admin/password", headers=headers, json={
            "currentPassword": self.changed, "newPassword": " " * 12,
        }).status_code, 422)

    def test_activity_dashboard_counts_real_messages_not_internal_audit(self):
        headers = self.admin()
        user = self.farmer()
        user["weather"] = {"enabled": True}
        self.farmers.save_user(user)
        other = self.farmer("254700000002")
        other["active"] = False
        self.farmers.save_user(other)
        self.containers["farms"].create_item({"id": "farm1", "owner": user["phoneNumber"]})
        for direction in ("inbound", "outbound"):
            self.containers["whatsapp_interactions"].create_item({
                "id": direction, "owner": user["phoneNumber"], "direction": direction,
                "type": "text", "timestamp": iso(utcnow()), "status": "stored",
                "message": "Do not include message bodies in dashboard previews.",
            })
        response = self.client.get("/admin/dashboard", headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        for key in ("farmers", "archivedFarmers", "farms", "registered", "weatherSubscribers",
                    "activeFarmers7Days", "incoming7Days", "outgoing7Days"):
            self.assertEqual(result[key], 1, key)
        self.assertEqual(sum(row["messages"] for row in result["daily"]), 1)
        self.assertNotIn("Do not include", response.text)

    def test_login_throttle_survives_service_recreation(self):
        for _ in range(15):
            self.assertEqual(self.login(password="incorrect").status_code, 401)
        self.assertEqual(self.login().status_code, 429)
        self.assertTrue(any(
            item.get("recordType") == "loginThrottle" for item in self.users.items.values()
        ))

    def test_managed_admin_can_take_over_without_phone_allowlist(self):
        headers = self.admin()
        user = self.farmer()
        path = f"/staff/conversations/{user['phoneNumber']}"
        current = self.client.get(path, headers=headers)
        self.assertEqual(current.status_code, 200, current.text)
        taken = self.client.put(path, headers=headers, json={
            "paused": True, "revision": current.json()["revision"],
        })
        self.assertEqual(taken.status_code, 200, taken.text)
        self.assertEqual(taken.json()["changedBy"], self.email)


if __name__ == "__main__":
    unittest.main()

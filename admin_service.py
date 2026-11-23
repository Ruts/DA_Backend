import hashlib
import hmac
import os
import secrets
from collections import Counter
from datetime import timedelta
from typing import Annotated, Literal
from uuid import uuid4

from azure.cosmos.exceptions import CosmosResourceExistsError
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from farmer_service import iso, parse_time, public_profile, utcnow
from meta_whatsapp import phone_id

PASSWORD_ITERATIONS = 600000
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
Password = Annotated[str, StringConstraints(
    min_length=12, max_length=256, strip_whitespace=False, pattern=r"\S",
)]


def password_hash(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest.hex()}"


def check_password(password, encoded):
    algorithm, iterations, salt, expected = encoded.split("$")
    if algorithm != "pbkdf2_sha256":
        raise ValueError("Unsupported password hash.")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations))
    return hmac.compare_digest(digest.hex(), expected)


def account_id(email):
    return "staff-" + hashlib.sha256(email.strip().lower().encode()).hexdigest()


class Values(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Login(Values):
    email: str = Field(max_length=254, pattern=EMAIL_PATTERN)
    password: str = Field(min_length=1, max_length=256)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class PasswordChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    currentPassword: str = Field(min_length=1, max_length=256)
    newPassword: Password


class StaffCreate(Values):
    email: str = Field(max_length=254, pattern=EMAIL_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    role: Literal["admin", "staff"] = "staff"
    password: Password
    stationId: str | None = None


class StaffUpdate(Values):
    name: str = Field(min_length=1, max_length=120)
    role: Literal["admin", "staff"]
    active: bool
    revision: str = Field(min_length=1, max_length=200)
    password: Password | None = None
    stationId: str | None = None


class FarmerCreate(Values):
    phoneNumber: str = Field(min_length=8, max_length=20)
    name: str = Field(min_length=1, max_length=120)
    language: Literal["en", "sw"] = "en"


class FarmerUpdate(Values):
    name: str = Field(min_length=1, max_length=120)
    language: Literal["en", "sw"]
    active: bool
    revision: str = Field(min_length=1, max_length=200)
    stationOverride: str | None = None


def public_account(account):
    return {
        **{key: account.get(key) for key in (
            "id", "email", "name", "role", "createdAt", "lastLoginAt", "deletedAt", "stationId",
        )},
        "active": account.get("active", True),
        "passwordChangeRequired": account.get("passwordChangeRequired", False),
        "bootstrap": account.get("bootstrap", False),
        "revision": account["_etag"],
    }


def public_farmer(user):
    return {
        **public_profile(user), "id": user["id"], "revision": user["_etag"],
        "active": user.get("active", True), "deletedAt": user.get("deletedAt"),
        "lastInboundAt": user.get("lastInboundAt"), "verified": bool(user.get("verified")),
        "stationOverride": user.get("stationOverride"), "primaryFarmId": user.get("primaryFarmId"),
    }


class AdminService:
    def __init__(self, farmers, create_token):
        self.farmers = farmers
        self.users = farmers.users
        self.create_token = create_token
        self.dummy_hash = password_hash(secrets.token_urlsafe(32))

    def find(self, record_id):
        records = list(self.users.query_items(
            "SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": record_id}], enable_cross_partition_query=True,
        ))
        return records[0] if records else None

    def bootstrap(self, email):
        configured_email = os.getenv("ADMIN_BOOTSTRAP_EMAIL", "").strip().lower()
        encoded = os.getenv("ADMIN_BOOTSTRAP_PASSWORD_HASH", "")
        if not configured_email or email != configured_email or not encoded:
            return None
        account = {
            "id": account_id(email), "recordType": "staffAccount", "email": email,
            "name": "Administrator", "role": "admin", "active": True, "bootstrap": True,
            "passwordHash": encoded, "passwordChangeRequired": True, "sessionVersion": str(uuid4()),
            "createdAt": iso(utcnow()),
        }
        try:
            return self.users.create_item(account)
        except CosmosResourceExistsError:
            return self.find(account["id"])

    def throttle(self, key):
        now = utcnow()
        record_id = "admin-login-" + hashlib.sha256(key.encode()).hexdigest()
        record = self.find(record_id)
        if not record:
            try:
                self.users.create_item({
                    "id": record_id, "recordType": "loginThrottle",
                    "attempts": 1, "windowAt": iso(now),
                })
                return
            except CosmosResourceExistsError:
                record = self.find(record_id)
        if now - parse_time(record["windowAt"]) >= timedelta(minutes=15):
            record.update(attempts=0, windowAt=iso(now))
        if record["attempts"] >= 15:
            raise HTTPException(429, "Too many sign-in attempts. Try again in 15 minutes.")
        record["attempts"] += 1
        self.farmers.save_user(record)

    def login(self, values, client_ip):
        email = values.email.strip().lower()
        # Both limits are persistent across API workers; ETags prevent lost increments.
        self.throttle("email:" + email)
        self.throttle("ip:" + client_ip)
        account = self.find(account_id(email)) or self.bootstrap(email)
        valid = check_password(values.password, account["passwordHash"] if account else self.dummy_hash)
        if not account or not valid or not account.get("active", True):
            raise HTTPException(401, "Invalid email or password.")
        account["lastLoginAt"] = iso(utcnow())
        self.farmers.save_user(account)
        return self.session(account)

    def session(self, account):
        token = self.create_token({
            "accountId": account["id"], "sessionVersion": account["sessionVersion"],
            "phone": account["email"], "name": account["name"],
        })
        return {"token": token, "account": public_account(account)}

    def validate_actor(self, payload):
        account = self.find(payload["accountId"])
        if (
            not account or account.get("recordType") != "staffAccount"
            or not account.get("active", True)
            or account["sessionVersion"] != payload.get("sessionVersion")
        ):
            raise HTTPException(401, "Staff session expired. Sign in again.")
        return {
            **payload, "phone": account["email"], "name": account["name"],
            "role": account["role"], "passwordChangeRequired": account["passwordChangeRequired"],
            "stationId": account.get("stationId"),
        }

    def require_account(self, actor, admin=False, allow_password_change=False):
        if not actor.get("accountId") or actor.get("role") not in {"admin", "staff"}:
            raise HTTPException(403, "Staff sign-in required.")
        if actor.get("passwordChangeRequired") and not allow_password_change:
            raise HTTPException(403, "Change your temporary password first.")
        if admin and actor["role"] != "admin":
            raise HTTPException(403, "Administrator access required.")
        if not allow_password_change and actor["role"] == "staff" and self.farmers.access:
            self.farmers.access.require_station(actor.get("stationId"))
        return self.find(actor["accountId"])

    def audit(self, actor, action, target):
        self.farmers.interactions.create_item({
            "id": str(uuid4()), "owner": target, "direction": "internal", "type": "administration",
            "senderType": "staff", "staffPhone": actor["phone"],
            "timestamp": iso(utcnow()), "status": "recorded", "message": action,
        })

    def page(self, query, cursor, limit, serializer, parameters=None):
        pages = self.users.query_items(
            query, parameters=parameters, enable_cross_partition_query=True, max_item_count=limit,
        ).by_page(continuation_token=cursor)
        return {
            "items": [serializer(item) for item in next(pages, [])],
            "cursor": pages.continuation_token,
        }

    def staff_account(self, record_id):
        account = self.find(record_id)
        if not account or account.get("recordType") != "staffAccount":
            raise HTTPException(404, "Staff account not found.")
        return account

    def check_revision(self, record, revision):
        if record["_etag"] != revision:
            raise HTTPException(409, "Record changed. Refresh before editing.")

    def dashboard(self, actor):
        clause, scope = self.farmers.access.scope(actor, "phoneNumber")
        farmers = list(self.users.query_items(
            "SELECT * FROM c WHERE IS_DEFINED(c.phoneNumber)" + (" AND " + clause if clause else ""),
            parameters=scope, enable_cross_partition_query=True,
        ))
        active = [u for u in farmers if u.get("active", True)]
        since = iso((utcnow() - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0))
        owner_clause, owner_scope = self.farmers.access.scope(actor, "owner")
        parameters = [{"name": "@since", "value": since}] + owner_scope
        activity = list(self.farmers.interactions.query_items(
            "SELECT c.owner, c.direction, c.timestamp, c.type FROM c WHERE c.timestamp >= @since "
            "AND (c.direction = 'inbound' OR c.direction = 'outbound')" +
            (" AND " + owner_clause if owner_clause else ""),
            parameters=parameters, enable_cross_partition_query=True,
        ))
        recent = list(self.farmers.interactions.query_items(
            "SELECT TOP 20 * FROM c" + (" WHERE " + owner_clause if owner_clause else "") +
            " ORDER BY c.timestamp DESC", parameters=owner_scope, enable_cross_partition_query=True,
        ))
        farms = list(self.farmers.farms.query_items(
            "SELECT VALUE COUNT(1) FROM c" + (" WHERE " + owner_clause if owner_clause else ""),
            parameters=owner_scope, enable_cross_partition_query=True,
        ))[0]
        inbound = [item for item in activity if item["direction"] == "inbound"]
        counts = Counter(item["timestamp"][:10] for item in inbound)
        return {
            "farmers": len(active), "archivedFarmers": len(farmers) - len(active),
            "farms": farms, "registered": sum(bool(u.get("verified")) for u in active),
            "weatherSubscribers": sum(bool(u.get("weather", {}).get("enabled")) for u in active),
            "activeFarmers7Days": len({item["owner"] for item in inbound}),
            "incoming7Days": len(inbound),
            "outgoing7Days": sum(item["direction"] == "outbound" for item in activity),
            "daily": [
                {"date": (utcnow() - timedelta(days=day)).date().isoformat(),
                 "messages": counts[(utcnow() - timedelta(days=day)).date().isoformat()]}
                for day in range(6, -1, -1)
            ],
            "recent": [
                {key: item.get(key) for key in (
                    "id", "owner", "direction", "type", "timestamp", "status", "senderType",
                )} for item in recent
            ],
        }

    def archive_farmer(self, user, actor):
        user.update(active=False, deletedAt=iso(utcnow()), authRevision=str(uuid4()))
        user.pop("codeHash", None)
        user["handoff"] = {
            "paused": True, "revision": str(uuid4()),
            "changedBy": actor["phone"], "changedAt": iso(utcnow()),
        }


def create_admin_router(service, current_user):
    router = APIRouter(prefix="/admin")

    def staff(actor=Depends(current_user)):
        service.require_account(actor)
        return actor

    def admin(actor=Depends(current_user)):
        service.require_account(actor, admin=True)
        return actor

    @router.post("/login")
    def login(values: Login, request: Request):
        return service.login(values, request.client.host if request.client else "unknown")

    @router.get("/session")
    def session(actor=Depends(current_user)):
        return public_account(service.require_account(actor, allow_password_change=True))

    @router.post("/password")
    def password(values: PasswordChange, actor=Depends(current_user)):
        account = service.require_account(actor, allow_password_change=True)
        service.throttle("password:" + account["id"])
        if not check_password(values.currentPassword, account["passwordHash"]):
            raise HTTPException(401, "Current password is incorrect.")
        if values.currentPassword == values.newPassword:
            raise HTTPException(422, "Choose a different password.")
        account.update(
            passwordHash=password_hash(values.newPassword), passwordChangeRequired=False,
            sessionVersion=str(uuid4()),
        )
        service.farmers.save_user(account)
        service.audit(actor, "Changed own password; previous sessions revoked.", account["id"])
        return service.session(account)

    @router.get("/dashboard")
    def dashboard(actor=Depends(staff)):
        return service.dashboard(actor)

    @router.get("/staff")
    def staff_list(
        cursor: str | None = Query(default=None, max_length=16384),
        limit: int = Query(default=50, ge=1, le=100), actor=Depends(admin),
    ):
        return service.page(
            "SELECT * FROM c WHERE c.recordType = 'staffAccount' ORDER BY c.name ASC",
            cursor, limit, public_account,
        )

    @router.post("/staff", status_code=201)
    def create_staff(values: StaffCreate, actor=Depends(admin)):
        if values.stationId:
            service.farmers.access.require_station(values.stationId)
        if values.role == "admin" and values.stationId:
            raise HTTPException(422, "Global administrators cannot be restricted to a station.")
        email = values.email.lower()
        if service.find(account_id(email)):
            raise HTTPException(409, "This email already exists. Edit or restore that staff account.")
        account = {
            "id": account_id(email), "recordType": "staffAccount", "email": email,
            "name": values.name, "role": values.role, "active": True,
            "passwordHash": password_hash(values.password), "passwordChangeRequired": True,
            "sessionVersion": str(uuid4()), "createdAt": iso(utcnow()),
            "stationId": values.stationId,
        }
        try:
            saved = service.users.create_item(account)
        except CosmosResourceExistsError:
            raise HTTPException(409, "This email already exists.")
        service.audit(actor, "Created staff account.", account["id"])
        return public_account(saved)

    @router.patch("/staff/{record_id}")
    def update_staff(record_id: str, values: StaffUpdate, actor=Depends(admin)):
        account = service.staff_account(record_id)
        service.check_revision(account, values.revision)
        if (account.get("bootstrap") or account["id"] == actor["accountId"]) and (
            values.role != account["role"] or not values.active
        ):
            raise HTTPException(409, "You cannot disable or demote yourself or the initial administrator.")
        if account["id"] == actor["accountId"] and values.password:
            raise HTTPException(409, "Use Change password for your own account.")
        station_id = values.stationId if "stationId" in values.model_fields_set else account.get("stationId")
        if values.role == "admin":
            station_id = None
        if station_id:
            service.farmers.access.require_station(station_id)
        account.update(name=values.name, role=values.role, active=values.active, stationId=station_id)
        if values.active:
            account.pop("deletedAt", None)
        if values.password:
            account.update(passwordHash=password_hash(values.password), passwordChangeRequired=True)
        if account["id"] != actor["accountId"]:
            account["sessionVersion"] = str(uuid4())
        service.farmers.save_user(account)
        service.audit(actor, "Updated staff account.", record_id)
        return public_account(account)

    @router.delete("/staff/{record_id}")
    def delete_staff(record_id: str, revision: str, actor=Depends(admin)):
        account = service.staff_account(record_id)
        service.check_revision(account, revision)
        if account.get("bootstrap") or account["id"] == actor["accountId"]:
            raise HTTPException(409, "You cannot archive yourself or the initial administrator.")
        account.update(active=False, deletedAt=iso(utcnow()), sessionVersion=str(uuid4()))
        service.farmers.save_user(account)
        service.audit(actor, "Archived staff account; sessions revoked.", record_id)
        return public_account(account)

    @router.get("/farmers")
    def farmer_list(
        cursor: str | None = Query(default=None, max_length=16384),
        limit: int = Query(default=50, ge=1, le=100), actor=Depends(staff),
    ):
        clause, parameters = service.farmers.access.scope(actor, "phoneNumber")
        page = service.page(
            "SELECT * FROM c WHERE IS_DEFINED(c.phoneNumber)" + (" AND " + clause if clause else "") +
            " ORDER BY c.name ASC", cursor, limit, public_farmer, parameters,
        )
        service.farmers.access.annotate_farmers(page["items"])
        return page

    @router.post("/farmers", status_code=201)
    def create_farmer(values: FarmerCreate, actor=Depends(admin)):
        try:
            phone = phone_id(values.phoneNumber)
        except ValueError as error:
            raise HTTPException(422, str(error))
        if service.farmers.find_user(phone):
            raise HTTPException(409, "This farmer already exists. Edit or restore the existing record.")
        user = {
            "id": phone, "phoneNumber": phone, "name": values.name, "language": values.language,
            "active": True, "verified": False, "onboardingStep": "consent", "createdAt": iso(utcnow()),
        }
        try:
            saved = service.users.create_item(user)
        except CosmosResourceExistsError:
            raise HTTPException(409, "This farmer already exists.")
        service.audit(actor, "Created farmer; WhatsApp consent and registration still required.", phone)
        return public_farmer(saved)

    @router.patch("/farmers/{phone}")
    def update_farmer(phone: str, values: FarmerUpdate, actor=Depends(staff)):
        user = service.farmers.authorised_user(actor, phone, include_inactive=True)
        service.check_revision(user, values.revision)
        if "stationOverride" in values.model_fields_set:
            if actor["role"] != "admin":
                raise HTTPException(403, "Only global administrators can override station assignments.")
            if values.stationOverride:
                service.farmers.access.require_station(values.stationOverride)
            user["stationOverride"] = values.stationOverride
        if not values.active:
            service.archive_farmer(user, actor)
        user.update(name=values.name, language=values.language, active=values.active)
        if values.active:
            user.pop("deletedAt", None)
        service.farmers.save_user(user)
        service.audit(actor, "Updated farmer profile.", user["phoneNumber"])
        return public_farmer(user)

    @router.delete("/farmers/{phone}")
    def delete_farmer(phone: str, revision: str, actor=Depends(staff)):
        user = service.farmers.authorised_user(actor, phone, include_inactive=True)
        service.check_revision(user, revision)
        service.archive_farmer(user, actor)
        service.farmers.save_user(user)
        service.audit(actor, "Archived farmer; login, AI and weather disabled. History retained.", user["phoneNumber"])
        return public_farmer(user)

    return router

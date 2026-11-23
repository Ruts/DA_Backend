import math
from collections import defaultdict
from typing import Literal
from uuid import uuid4

from azure.cosmos.exceptions import CosmosResourceExistsError
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from farmer_service import iso, public_farm, utcnow
from meta_whatsapp import phone_id

SERVICES = {
    "weather": ("Weather forecasts", True),
    "chat": ("Chat history", True),
    "reports": ("Farm reports", True),
    "drones": ("Drone services", False),
}


class StationValues(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=120, pattern=r"\S")
    locationName: str = Field(default="", max_length=200)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radiusKm: float = Field(gt=0, le=1000)
    contact: str = Field(default="", max_length=200)
    active: bool = True
    revision: str = Field(default="", max_length=200)


class ServiceValues(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    scope: Literal["all", "stations"]
    stationIds: list[str] = Field(default_factory=list, max_length=100)
    revision: str = Field(default="", max_length=200)


class PrimaryFarm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    farmId: str = Field(min_length=1, max_length=200)


def farm_point(farm):
    lat, lon = farm.get("latitude"), farm.get("longitude")
    if lat is None or lon is None:
        points = farm.get("coordinates") or []
        if not points:
            return None
        if any(not isinstance(p, dict) or not isinstance(p.get("lat"), (int, float))
               or not isinstance(p.get("lng"), (int, float)) for p in points):
            return None
        lat = sum(p["lat"] for p in points) / len(points)
        lon = sum(p["lng"] for p in points) / len(points)
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        return None
    if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def distance_km(point, station):
    lat1, lon1, lat2, lon2 = map(math.radians, (*point, station["latitude"], station["longitude"]))
    value = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1, math.sqrt(value)))


class StationService:
    def __init__(self, farmers):
        self.farmers = farmers
        self.users = farmers.users

    def find(self, record_id):
        items = list(self.users.query_items(
            "SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": record_id}], enable_cross_partition_query=True,
        ))
        return items[0] if items else None

    def stations(self):
        return list(self.users.query_items(
            "SELECT * FROM c WHERE c.recordType = @recordType ORDER BY c.name ASC",
            parameters=[{"name": "@recordType", "value": "station"}], enable_cross_partition_query=True,
        ))

    def public_station(self, station):
        return {key: value for key, value in station.items() if not key.startswith("_") and key != "recordType"} | {
            "revision": station["_etag"],
        }

    def require_station(self, station_id):
        station = self.find(station_id) if station_id else None
        if not station or station.get("recordType") != "station" or not station.get("active"):
            raise HTTPException(403, "An active station assignment is required. Contact an administrator.")
        return station

    def covering(self, farm, stations):
        point = farm_point(farm)
        if point is None:
            return []
        matches = [(distance_km(point, s), s) for s in stations if s.get("active")]
        return [s for distance, s in sorted(matches, key=lambda match: (match[0], match[1]["id"]))
                if distance <= s["radiusKm"]]

    def assignment(self, user, farms=None, stations=None):
        stations = self.stations() if stations is None else stations
        if user.get("stationOverride"):
            return next((s for s in stations if s["id"] == user["stationOverride"] and s.get("active")), None)
        farms = self.farmers.farm_list(user) if farms is None else farms
        farms = sorted(farms, key=lambda f: f["id"])
        primary = next((f for f in farms if f["id"] == user.get("primaryFarmId")), farms[0] if farms else None)
        matches = self.covering(primary, stations) if primary else []
        return matches[0] if matches else None

    def owners(self, actor):
        if actor.get("accountId") and actor.get("role") == "admin":
            return None
        station = self.require_station(actor.get("stationId"))
        stations = self.stations()
        farms_by_owner = self.farm_index()
        owners = []
        for user in self.users.query_items(
            "SELECT * FROM c WHERE IS_DEFINED(c.phoneNumber)", enable_cross_partition_query=True,
        ):
            assigned = self.assignment(user, farms_by_owner[phone_id(user["phoneNumber"])], stations)
            if assigned and assigned["id"] == station["id"]:
                owners.append(phone_id(user["phoneNumber"]))
        return owners

    def farm_index(self, owners=None):
        parameters = [] if owners is None else [{
            "name": "@owners", "value": owners + ["+" + p for p in owners],
        }]
        query = "SELECT * FROM c" + (" WHERE ARRAY_CONTAINS(@owners, c.owner)" if owners is not None else "")
        grouped = defaultdict(list)
        for farm in self.farmers.farms.query_items(query, parameters=parameters, enable_cross_partition_query=True):
            grouped[phone_id(farm["owner"])].append(farm)
        return grouped

    def annotate_farmers(self, records):
        stations = self.stations()
        farms = self.farm_index([phone_id(u["phoneNumber"]) for u in records])
        for user in records:
            assigned = self.assignment(user, farms[phone_id(user["phoneNumber"])], stations)
            user["stationName"] = assigned["name"] if assigned else None
        return records

    def authorise(self, actor, user):
        if actor.get("accountId") and actor.get("role") == "admin":
            return
        station = self.require_station(actor.get("stationId"))
        assigned = self.assignment(user)
        if not assigned or assigned["id"] != station["id"]:
            raise HTTPException(403, "This farmer is outside your station.")

    def scope(self, actor, field):
        owners = self.owners(actor)
        if owners is None:
            return "", []
        return f"ARRAY_CONTAINS(@owners, c.{field})", [{
            "name": "@owners", "value": owners + ["+" + p for p in owners],
        }]

    def rules(self):
        record = self.find("farmer-service-config")
        saved = (record or {}).get("rules", {})
        return {
            "revision": record["_etag"] if record else "",
            "items": [
                {"id": key, "name": name, **saved.get(key, {
                    "enabled": default, "scope": "stations" if key == "drones" else "all", "stationIds": [],
                })}
                for key, (name, default) in SERVICES.items()
            ],
        }

    def available(self, key, farm, stations=None, rules=None):
        rule = next(r for r in (rules or self.rules())["items"] if r["id"] == key)
        if not rule["enabled"]:
            return False
        if rule["scope"] == "all" and key != "drones":
            return True
        covered = self.covering(farm, self.stations() if stations is None else stations)
        return any(s["id"] in rule["stationIds"] for s in covered)

    def require_feature(self, actor, key, farm_id=None):
        user = self.farmers.authorised_user(actor, actor["phone"])
        farms = self.farmers.farm_list(user)
        if farm_id:
            farms = [self.farmers.farm(user, farm_id)]
        rules, stations = self.rules(), self.stations()
        if not any(self.available(key, f, stations, rules) for f in farms or [{}]):
            raise HTTPException(403, "This service is not available for your farm.")

    def allowed_farms(self, actor, key):
        user = self.farmers.authorised_user(actor, actor["phone"])
        stations, rules = self.stations(), self.rules()
        return {f["id"] for f in self.farmers.farm_list(user) if self.available(key, f, stations, rules)}

    def snapshot(self, user):
        farms = self.farmers.farm_list(user)
        stations, rules = self.stations(), self.rules()
        assigned = self.assignment(user, farms, stations)
        primary = user.get("primaryFarmId") or (min(farms, key=lambda f: f["id"])["id"] if farms else None)
        drones = next(r for r in rules["items"] if r["id"] == "drones")
        return {
            "primaryFarmId": primary,
            "station": self.public_station(assigned) if assigned else None,
            "services": [r["id"] for r in rules["items"]
                         if any(self.available(r["id"], f, stations, rules) for f in farms or [{}])],
            "farms": [{
                **public_farm(farm),
                "services": [r["id"] for r in rules["items"] if self.available(r["id"], farm, stations, rules)],
                "stations": [self.public_station(s) for s in self.covering(farm, stations)],
                "droneStations": [self.public_station(s) for s in self.covering(farm, stations)
                                  if drones["enabled"] and s["id"] in drones["stationIds"]],
            } for farm in farms],
        }


def create_station_router(service, admins, current_user):
    router = APIRouter()

    def admin(actor=Depends(current_user)):
        admins.require_account(actor, admin=True)
        return actor

    @router.get("/admin/stations")
    def stations(actor=Depends(current_user)):
        admins.require_account(actor)
        records = service.stations()
        if actor["role"] != "admin":
            records = [s for s in records if s["id"] == actor.get("stationId")]
        return {"items": [service.public_station(s) for s in records]}

    @router.post("/admin/stations", status_code=201)
    def create(values: StationValues, actor=Depends(admin)):
        record = {
            **values.model_dump(exclude={"revision"}), "id": "station-" + str(uuid4()),
            "recordType": "station", "createdAt": iso(utcnow()),
        }
        saved = service.users.create_item(record)
        admins.audit(actor, "Created station.", saved["id"])
        return service.public_station(saved)

    @router.patch("/admin/stations/{station_id}")
    def update(station_id: str, values: StationValues, actor=Depends(admin)):
        station = service.find(station_id)
        if not station or station.get("recordType") != "station":
            raise HTTPException(404, "Station not found.")
        admins.check_revision(station, values.revision)
        station.update(values.model_dump(exclude={"revision"}), updatedAt=iso(utcnow()))
        service.farmers.save_user(station)
        admins.audit(actor, "Updated station and coverage.", station_id)
        return service.public_station(station)

    @router.get("/admin/services")
    def rules(actor=Depends(admin)):
        return service.rules()

    @router.put("/admin/services/{key}")
    def update_rule(key: str, values: ServiceValues, actor=Depends(admin)):
        if key not in SERVICES:
            raise HTTPException(404, "Service not found.")
        if key == "drones" and values.scope != "stations":
            raise HTTPException(422, "Drone services must be restricted to station coverage.")
        if values.enabled and values.scope == "stations" and not values.stationIds:
            raise HTTPException(422, "Choose at least one active station.")
        selected = [s for s in service.stations() if s["id"] in values.stationIds]
        if len(selected) != len(set(values.stationIds)):
            raise HTTPException(422, "A selected station no longer exists.")
        if values.enabled and values.scope == "stations" and not any(s.get("active") for s in selected):
            raise HTTPException(422, "Choose at least one active station.")
        record = service.find("farmer-service-config")
        if record:
            admins.check_revision(record, values.revision)
        elif values.revision:
            raise HTTPException(409, "Service settings changed. Refresh and retry.")
        else:
            record = {"id": "farmer-service-config", "recordType": "serviceConfig", "rules": {}}
        record["rules"][key] = values.model_dump(exclude={"revision"})
        if "_etag" in record:
            service.farmers.save_user(record)
        else:
            try:
                service.users.create_item(record)
            except CosmosResourceExistsError:
                raise HTTPException(409, "Service settings changed. Refresh and retry.")
        admins.audit(actor, "Updated farmer service: " + key, record["id"])
        return service.rules()

    @router.get("/farmer-services")
    def farmer_services(actor=Depends(current_user)):
        return service.snapshot(service.farmers.authorised_user(actor, actor["phone"]))

    @router.put("/profiles/{phone}/primary-farm")
    def primary_farm(phone: str, values: PrimaryFarm, actor=Depends(current_user)):
        user = service.farmers.authorised_user(actor, phone)
        service.farmers.farm(user, values.farmId)
        if actor.get("accountId") and actor.get("role") != "admin":
            raise HTTPException(403, "Only the farmer or a global administrator can change the primary farm.")
        user["primaryFarmId"] = values.farmId
        service.farmers.save_user(user)
        return service.snapshot(user)

    return router

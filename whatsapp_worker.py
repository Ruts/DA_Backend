import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone

from azure.core import MatchConditions
from azure.cosmos.exceptions import CosmosAccessConditionFailedError, CosmosHttpResponseError
from fastapi import HTTPException

from farmer_service import (
    automation_blocked, automation_paused, automation_revision, iso, next_forecast, utcnow,
)
from meta_whatsapp import phone_id, required

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 5


class WhatsAppWorker:
    def __init__(self, service):
        self.service = service

    def schedule(self):
        now = utcnow()
        users = self.service.users.query_items(
            "SELECT * FROM c WHERE c.weather.enabled = true AND c.weather.nextDueAt <= @now",
            parameters=[{"name": "@now", "value": iso(now)}],
            enable_cross_partition_query=True,
        )
        for user in users:
            if automation_paused(user):
                continue
            if self.service.access:
                farm = self.service.farm(user, user["weather"]["farmId"])
                if not self.service.access.available("weather", farm):
                    continue
            settings = user["weather"]
            due = settings["nextDueAt"]
            key = f"{user['id']}:{settings['revision']}:{due}"
            self.service.enqueue({
                "id": hashlib.sha256(key.encode()).hexdigest(),
                "owner": phone_id(user["phoneNumber"]), "kind": "weather",
                "revision": settings["revision"], "status": "pending",
                "createdAt": iso(now), "availableAt": iso(now),
            })
            settings["nextDueAt"] = next_forecast(settings, now)
            try:
                self.service.save_user(user)
            except CosmosAccessConditionFailedError:
                logger.info("Weather settings changed while scheduling; rechecking next tick.")

    def save_job(self, job):
        updated = self.service.jobs.replace_item(
            item=job["id"], body=job, etag=job["_etag"],
            match_condition=MatchConditions.IfNotModified,
        )
        job.update(updated)

    def prepare_inbound(self, job):
        service = self.service
        message = job["message"]
        service.record_inbound(message)
        user = service.find_user(job["owner"])
        if not user:
            user = service.users.create_item({
                "id": job["owner"], "phoneNumber": job["owner"], "name": "",
                "language": "en", "verified": False, "onboardingStep": "consent",
            })
        job["automationRevision"] = automation_revision(user)
        if user.get("active") is False:
            return None
        if user.get("lastHandledId") == message["id"]:
            return user["lastReply"]
        user["lastInboundAt"] = iso(
            datetime.fromtimestamp(int(message["timestamp"]), timezone.utc)
            if message.get("timestamp") else utcnow()
        )
        if automation_blocked(user, job):
            text = message.get("text", {}).get("body", "").strip().upper()
            if text in {"STOP", "STOP WEATHER", "ACHA", "SIMAMISHA"}:
                service.handle(user, message)
            farm = service.active_farm(user)
            if farm and message["type"] in {"text", "image"}:
                details = message.get(message["type"], {})
                service.save_chat(
                    user, farm, "user", details.get("body") or details.get("caption") or "[Photo]",
                    message["id"], media_id=details.get("id") if message["type"] == "image" else None,
                    timestamp=user["lastInboundAt"],
                )
            reply = None
        else:
            try:
                reply = service.handle(user, message)
            except (ValueError, HTTPException) as error:
                reply = error.detail if isinstance(error, HTTPException) else str(error)
        user["lastHandledId"], user["lastReply"] = message["id"], reply
        service.save_user(user)
        return reply

    def deliver(self, job):
        service = self.service
        if job["kind"] == "inbound":
            if "reply" not in job:
                job["reply"] = self.prepare_inbound(job)
                self.save_job(job)
            user = service.find_user(job["owner"])
            if (
                not job["reply"] or automation_blocked(user, job)
                or automation_revision(user) != job.get("automationRevision", "")
            ):
                return None
            job["status"] = "sending"
            self.save_job(job)
            return service.send_recorded(
                job["owner"], job["reply"],
                lambda: service.messenger.text(job["owner"], job["reply"]), key=job["id"],
            )
        user = service.find_user(job["owner"])
        settings = user.get("weather", {}) if user else {}
        if (
            automation_blocked(user, job)
            or not settings.get("enabled") or settings.get("revision") != job["revision"]
        ):
            return None
        revision = automation_revision(user)
        language = user.get("language", "en")
        farm = service.farm(user, settings["farmId"])
        if service.access and not service.access.available("weather", farm):
            return None
        forecast = service.forecast(farm, language)
        template = required("META_WEATHER_TEMPLATE_SW" if language == "sw" else "META_WEATHER_TEMPLATE_EN")
        # Re-read immediately before sending, so queued forecasts respect an opt-out or edit.
        latest_user = service.find_user(job["owner"])
        latest = latest_user.get("weather", {}) if latest_user else {}
        if (
            automation_blocked(latest_user, job) or automation_revision(latest_user) != revision
            or not latest.get("enabled") or latest.get("revision") != job["revision"]
        ):
            return None
        if service.access and not service.access.available("weather", service.farm(latest_user, latest["farmId"])):
            return None
        job["status"] = "sending"
        self.save_job(job)
        return service.send_recorded(
            job["owner"], forecast,
            lambda: service.messenger.template(
                job["owner"], template, "sw" if language == "sw" else "en",
                [forecast.replace("\n", " ")],
            ), key=job["id"],
        )

    def tick(self):
        self.schedule()
        now = utcnow()
        jobs = self.service.jobs.query_items(
            "SELECT * FROM c WHERE "
            "(c.status = 'pending' AND c.availableAt <= @now) OR "
            "(c.status = 'processing' AND c.leaseUntil <= @now) "
            "ORDER BY c.createdAt ASC",
            parameters=[{"name": "@now", "value": iso(now)}],
            enable_cross_partition_query=True,
        )
        processed, blocked_owners = 0, set()
        for job in jobs:
            if processed >= 50:
                break
            if job["kind"] == "inbound":
                if job["owner"] in blocked_owners:
                    continue
                user = self.service.find_user(job["owner"])
                since = (user or {}).get("handoff", {}).get("changedAt", "0001-01-01T00:00:00+00:00")
                earlier = list(self.service.jobs.query_items(
                    "SELECT TOP 1 c.id FROM c WHERE c.owner = @owner AND c.kind = 'inbound' "
                    "AND c.createdAt < @created AND c.createdAt > @since AND "
                    "(c.status = 'pending' OR c.status = 'processing' OR c.status = 'failed')",
                    parameters=[
                        {"name": "@owner", "value": job["owner"]},
                        {"name": "@created", "value": job["createdAt"]},
                        {"name": "@since", "value": since},
                    ], enable_cross_partition_query=True,
                ))
                if earlier:
                    blocked_owners.add(job["owner"])
                    continue
            job["status"] = "processing"
            job["leaseUntil"] = iso(utcnow() + timedelta(minutes=10))
            job["attempts"] = job.get("attempts", 0) + 1
            try:
                self.save_job(job)
            except CosmosAccessConditionFailedError:
                continue
            processed += 1
            try:
                message_id = self.deliver(job)
            except Exception:
                # The worker boundary must preserve failed jobs without logging private payloads.
                logger.exception("WhatsApp job %s failed", job["id"])
                job["lastErrorAt"] = iso(utcnow())
                job["status"] = (
                    "delivery_unknown" if job["status"] == "sending"
                    else "failed" if job["attempts"] >= 5 else "pending"
                )
                job["availableAt"] = iso(utcnow() + timedelta(minutes=min(30, 2 ** job["attempts"])))
                self.save_job(job)
                continue
            job["status"] = "sent" if message_id else "cancelled"
            job["metaMessageId"] = message_id
            job["completedAt"] = iso(utcnow())
            job["ttl"] = 8 * 86400
            self.save_job(job)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    from main import farmer_service

    logging.basicConfig(level=logging.INFO)
    worker = WhatsAppWorker(farmer_service)
    while True:
        try:
            worker.tick()
        except CosmosHttpResponseError:
            logger.exception("Cosmos worker tick failed; retrying in %s seconds.", POLL_INTERVAL_SECONDS)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

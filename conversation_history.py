import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceExistsError
from fastapi import APIRouter, Depends, HTTPException, Query
from jose import JWTError, jwt

from meta_whatsapp import phone_id, required

logger = logging.getLogger(__name__)


def chat_owner(chat):
    owner = chat["user"]
    return phone_id(owner["phone"] if isinstance(owner, dict) else owner)


def record_portal_chat(interactions, chat):
    timestamp = datetime.fromisoformat(chat["timestamp"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    record = {
        "id": "portal-" + hashlib.sha256(chat["id"].encode()).hexdigest(),
        "owner": chat_owner(chat), "direction": "inbound" if chat["role"] == "user" else "outbound",
        "message": chat["message"], "timestamp": timestamp.astimezone(timezone.utc).isoformat(),
        "channel": "portal", "legacyChatId": chat["id"], "type": "text", "status": "stored",
    }
    if chat.get("farmId"):
        record["farmId"] = chat["farmId"]
    try:
        interactions.create_item(record)
    except CosmosResourceExistsError:
        return False
    return True


def backfill_portal_history(chats, interactions):
    whatsapp_copies = set()
    for record in interactions.query_items(
        "SELECT c.id, c.owner, c.direction, c.metaMessageId, c.senderType FROM c",
        enable_cross_partition_query=True,
    ):
        if record.get("direction") not in {"inbound", "outbound"}:
            continue
        owner = phone_id(record["owner"])
        if record.get("direction") == "inbound" and record.get("metaMessageId"):
            for role in ("user", "assistant"):
                whatsapp_copies.add((owner, hashlib.sha256(f"{record['metaMessageId']}:{role}".encode()).hexdigest()))
        if record.get("senderType") == "staff":
            whatsapp_copies.add((owner, hashlib.sha256(f"{record['id']}:assistant".encode()).hexdigest()))
    counts = {"imported": 0, "alreadyImported": 0, "whatsappAdviceCopiesSkipped": 0, "invalidOwnersSkipped": 0}
    for chat in chats.query_items("SELECT * FROM c", enable_cross_partition_query=True):
        try:
            owner = chat_owner(chat)
        except (KeyError, ValueError):
            logger.warning("Legacy history has no valid phone owner; original record retained.")
            counts["invalidOwnersSkipped"] += 1
            continue
        if chat.get("channel") == "whatsapp" or (owner, chat["id"]) in whatsapp_copies:
            counts["whatsappAdviceCopiesSkipped"] += 1
            continue
        counts["imported" if record_portal_chat(interactions, chat) else "alreadyImported"] += 1
    return counts


def create_history_router(service, current_user):
    router = APIRouter()

    @router.get("/chat-history/page")
    def history(
        cursor: str | None = Query(default=None, max_length=16384),
        limit: int = Query(default=25, ge=1, le=50), actor=Depends(current_user),
    ):
        service.access.require_feature(actor, "chat")
        owner = phone_id(actor["phone"])
        rule = next(r for r in service.access.rules()["items"] if r["id"] == "chat")
        farms = sorted(service.access.allowed_farms(actor, "chat")) if rule["scope"] != "all" else None
        context = hashlib.sha256(json.dumps([owner, farms], separators=(",", ":")).encode()).hexdigest()
        token = None
        through = datetime.now(timezone.utc).isoformat()
        if cursor:
            try:
                claims = jwt.decode(cursor, required("TOKEN_SECRET_KEY"), algorithms=[required("TOKEN_ALGORITHM")])
            except JWTError as error:
                raise HTTPException(400, "History cursor is invalid or expired. Refresh the conversation.") from error
            if claims.get("purpose") != "chat-page" or claims.get("context") != context:
                raise HTTPException(409, "History access changed. Refresh the conversation.")
            token, through = claims["continuation"], claims["through"]
        parameters = [
            {"name": "@phone", "value": owner}, {"name": "@through", "value": through},
        ]
        query = (
            "SELECT * FROM c WHERE c.owner = @phone AND c.timestamp <= @through "
            "AND (c.direction = 'inbound' OR c.direction = 'outbound') "
            "AND (c.status = 'received' OR c.status = 'accepted' OR c.status = 'delivery_unknown' OR c.status = 'stored')"
        )
        if farms is not None:
            query += " AND ARRAY_CONTAINS(@farms, c.farmId)"
            parameters.append({"name": "@farms", "value": farms})
        try:
            pages = service.interactions.query_items(
                query + " ORDER BY c.timestamp DESC", parameters=parameters,
                partition_key=owner, max_item_count=limit,
            ).by_page(continuation_token=token)
            records = list(next(pages, []))
        except CosmosHttpResponseError as error:
            if error.status_code != 400:
                raise
            logger.warning("Conversation history continuation rejected by storage.")
            raise HTTPException(400, "History page expired. Refresh the conversation.") from error
        continuation = pages.continuation_token
        next_cursor = jwt.encode({
            "purpose": "chat-page", "context": context, "through": through, "continuation": continuation,
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
        }, required("TOKEN_SECRET_KEY"), algorithm=required("TOKEN_ALGORITHM")) if continuation else None
        return {
            "items": [{
                "id": item["id"], "role": "user" if item["direction"] == "inbound" else "assistant",
                "message": item["message"], "timestamp": item["timestamp"], "status": item["status"],
                "channel": item.get("channel", "whatsapp"), "type": item.get("type", "text"),
                "senderType": item.get("senderType"),
            } for item in reversed(records)],
            "cursor": next_cursor,
        }

    return router


if __name__ == "__main__":
    import argparse
    from azure.cosmos import CosmosClient
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Copy legacy portal history without deleting or resending messages.")
    parser.add_argument("--backfill", action="store_true", required=True)
    parser.parse_args()
    load_dotenv()
    database = CosmosClient(required("COSMOS_ENDPOINT"), required("COSMOS_KEY")).get_database_client(required("DATABASE_NAME"))
    print(json.dumps(backfill_portal_history(database.get_container_client("chats"),
                                           database.get_container_client("whatsapp_interactions"))))

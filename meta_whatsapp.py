import hashlib
import hmac
import os
import re
from urllib.parse import urlparse

import requests


def phone_id(value):
    if not isinstance(value, str):
        raise ValueError("Use an international phone number including country code.")
    value = re.sub(r"[\s()+-]", "", value)
    if not re.fullmatch(r"[1-9][0-9]{7,14}", value):
        raise ValueError("Use an international phone number including country code.")
    return value


def required(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing deployment setting: {name}")
    return value


def valid_signature(body, signature):
    expected = "sha256=" + hmac.new(
        required("META_APP_SECRET").encode(), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature or "")


class MetaWhatsApp:
    def graph(self, path, payload=None):
        version = required("META_GRAPH_VERSION")
        response = requests.request(
            "GET" if payload is None else "POST",
            f"https://graph.facebook.com/{version}/{path}",
            headers={"Authorization": f"Bearer {required('META_ACCESS_TOKEN')}"},
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"Meta API failed with HTTP {response.status_code}")
        return response.json()

    def send(self, phone, message):
        result = self.graph(f"{required('META_PHONE_NUMBER_ID')}/messages", {
            "messaging_product": "whatsapp",
            "to": phone_id(phone),
            **message,
        })
        return result["messages"][0]["id"]

    def text(self, phone, text):
        return self.send(phone, {"type": "text", "text": {"body": text[:4096]}})

    def template(self, phone, name, language, parameters, authentication=False):
        components = [{
            "type": "body",
            "parameters": [{"type": "text", "text": str(p)} for p in parameters],
        }]
        if authentication:
            components.append({
                "type": "button", "sub_type": "url", "index": "0",
                "parameters": [{"type": "text", "text": str(parameters[0])}],
            })
        return self.send(phone, {
            "type": "template",
            "template": {
                "name": name, "language": {"code": language},
                "components": components,
            },
        })

    def image(self, media_id):
        metadata = self.graph(media_id)
        if metadata.get("mime_type") not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError("Please send a JPEG, PNG or WebP photo.")
        if metadata.get("file_size", 0) > 5 * 1024 * 1024:
            raise ValueError("Please send a photo smaller than 5 MB.")
        url = metadata["url"]
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not any(
            host == domain or host.endswith("." + domain)
            for domain in ("facebook.com", "fbcdn.net", "fbsbx.com")
        ):
            raise ValueError("Meta returned an unexpected media host.")
        with requests.get(
            url, headers={"Authorization": f"Bearer {required('META_ACCESS_TOKEN')}"},
            timeout=30, stream=True, allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Meta media download failed: HTTP {response.status_code}")
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > 5 * 1024 * 1024:
                    raise ValueError("Please send a photo smaller than 5 MB.")
        return bytes(content), metadata["mime_type"]

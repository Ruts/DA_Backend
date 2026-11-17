import argparse

from dotenv import load_dotenv

from meta_whatsapp import MetaWhatsApp, phone_id, required


def main():
    parser = argparse.ArgumentParser(description="Configure a direct Meta WhatsApp Cloud API sender.")
    parser.add_argument("action", choices=["check", "register", "subscribe"])
    args = parser.parse_args()
    load_dotenv()
    api = MetaWhatsApp()
    if args.action == "check":
        details = api.graph(
            f"{required('META_PHONE_NUMBER_ID')}?fields=display_phone_number,verified_name,code_verification_status"
        )
        if phone_id(details["display_phone_number"]) != phone_id(required("WHATSAPP_BUSINESS_NUMBER")):
            raise RuntimeError("The Meta phone-number ID does not match the configured service number.")
        print(f"Number: {details['display_phone_number']}")
        print(f"Business: {details.get('verified_name', 'pending')}")
        print(f"Verification: {details.get('code_verification_status', 'unknown')}")
    elif args.action == "register":
        pin = required("META_REGISTRATION_PIN")
        if len(pin) != 6 or not pin.isascii() or not pin.isdigit():
            raise ValueError("META_REGISTRATION_PIN must contain six digits.")
        result = api.graph(f"{required('META_PHONE_NUMBER_ID')}/register", {
            "messaging_product": "whatsapp", "pin": pin,
        })
        if not result.get("success"):
            raise RuntimeError("Meta did not confirm phone registration.")
        print("Phone registered. Remove META_REGISTRATION_PIN from the process environment and retain it securely.")
    else:
        result = api.graph(f"{required('META_WABA_ID')}/subscribed_apps", {})
        if not result.get("success"):
            raise RuntimeError("Meta did not confirm the app subscription.")
        print("App subscribed to the WhatsApp Business Account.")


if __name__ == "__main__":
    main()

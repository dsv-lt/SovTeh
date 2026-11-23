"""Изолированный клиент для тестирования REST API агента (с обходом прокси Windows)."""

from __future__ import annotations

import json
import sys

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://localhost:8000/ask"

PAYLOAD = {
    "question": "Найди сюжет про космическую экспедицию к Марсу и расскажи, что там происходит",
    "k": 3,
}


def main() -> None:
    print("Отправка тестового запроса к ИИ-агенту через localhost...")
    try:
        session = requests.Session()
        session.trust_env = False  # Жёсткое отключение прокси-переменных Windows

        response = session.post(URL, json=PAYLOAD, timeout=60)
        response.raise_for_status()
        data = response.json()

        print("\nОтвет ИИ-агента:")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"\nОшибка при отправке запроса: {exc}")


if __name__ == "__main__":
    main()

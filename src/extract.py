import json
import time

import requests

from config import DEPARTEMENTS, OPENAGENDA_API, OPENAGENDA_KEY, RAW_FILE


def fetch_agendas() -> list[dict]:
    agendas = {}

    for department in DEPARTEMENTS:
        after = None

        while True:
            params = {
                "search": department,
                "size": 100,
                "sort": "recentlyAddedEvents.desc",
            }
            if after:
                params["after[]"] = after

            response = requests.get(
                OPENAGENDA_API,
                headers={"key": OPENAGENDA_KEY},
                params=params,
                timeout=30,
            )
            if not response.ok:
                raise RuntimeError(
                    f"OpenAgenda HTTP {response.status_code}: {response.text}"
                )

            data = response.json()
            page = data.get("agendas")
            if page is None:
                raise RuntimeError(f"Réponse OpenAgenda inattendue : {data}")

            for agenda in page:
                if agenda.get("uid") is not None:
                    agendas[agenda["uid"]] = agenda

            print(f"[extract] {department}: {len(agendas)} agendas uniques")
            after = data.get("after")
            if not after:
                break

            time.sleep(0.1)

    return list(agendas.values())


if __name__ == "__main__":
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(fetch_agendas(), ensure_ascii=False), encoding="utf-8")
    print(f"[extract] agendas sauvegardés : {RAW_FILE}")
from typing import List

from deal_daddy.settings import settings

MODELS_MODULES: List[str] = ["deal_daddy.db.models.dummy_model"]  # noqa: WPS407

TORTOISE_CONFIG = {  # noqa: WPS407
    "connections": {
        "default": str(settings.db_url),
    },
    "apps": {
        "models": {
            "models": MODELS_MODULES + ["aerich.models"],
            "default_connection": "default",
        },
    },
}

import os
from typing import Any, Dict, Optional, Sequence

from stream_platform.backend.models.load_model import build_movinet_a0_stream
from stream_platform.backend.models.model_infer import download_model
from stream_platform.backend.src.camera_loop import capture_loop
from stream_platform.backend.utils.logger import get_logger, setup_logging

NUM_CLASSES = 2
DEFAULT_CLASS_NAMES = ["no_violence", "violence"]


def _extract_class_names(meta: Optional[Dict[str, Any]]) -> Optional[Sequence[str]]:
    if not meta:
        return None
    if "class_names" in meta:
        names = meta["class_names"]
        if isinstance(names, (list, tuple)):
            return list(names)
    if "idx_to_class" in meta:
        mapping = meta["idx_to_class"]
        if isinstance(mapping, dict):
            try:
                return [mapping[i] for i in sorted(mapping)]
            except Exception:
                return list(mapping.values())
        if isinstance(mapping, (list, tuple)):
            return list(mapping)
    return None


def main():
    setup_logging(
        app_name="stream_platform",
        level="DEBUG" if os.getenv("ENV") == "development" else "INFO",
        json=False,
    )
    log = get_logger("movinet")
    log.info("Stream platform started")

    model = build_movinet_a0_stream(NUM_CLASSES)
    model, meta = download_model(model=model)

    class_names = _extract_class_names(meta) or DEFAULT_CLASS_NAMES
    positive_label = os.getenv("POSITIVE_LABEL", "violence")

    capture_loop(
        model=model,
        class_names=class_names,
        positive_label=positive_label,
    )


if __name__ == "__main__":
    main()

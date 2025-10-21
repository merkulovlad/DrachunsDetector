import logging, os, sys, structlog
from logging.handlers import RotatingFileHandler

def setup_logging(app_name: str = "stream_platform",
                  level: str | int = None,
                  json: bool = False,
                  log_file: str | None = None):
    level = level or os.getenv("LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # ---- stdlib logging sinks (console + optional file)
    handlers = []

    console = logging.StreamHandler(sys.stdout)
    if json:
        from pythonjsonlogger import jsonlogger
        console.setFormatter(jsonlogger.JsonFormatter("%(message)s"))
    else:
        console.setFormatter(logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    handlers.append(console)

    if log_file:
        fileh = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=3)
        fileh.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(fileh)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)

    timestamper = structlog.processors.TimeStamper(fmt="iso", key="ts")
    shared = {"app": app_name, "pid": os.getpid()}

    structlog.configure(
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.contextvars.merge_contextvars,     # support contextvars
            structlog.processors.add_log_level,          # level field
            timestamper,                                 # ts field
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            (structlog.processors.JSONRenderer() if json
             else structlog.dev.ConsoleRenderer(colors=True)),
        ],
    )

    # return a pre-bound logger with shared fields
    return structlog.get_logger().bind(**shared)

# convenience accessor
def get_logger(name: str = None):
    logger = structlog.get_logger()
    return logger if name is None else logger.bind(logger=name)

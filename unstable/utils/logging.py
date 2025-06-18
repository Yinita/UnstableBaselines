import logging, os, pathlib
from rich.logging import RichHandler

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

def setup_logger(name: str, log_dir: str, level: int = logging.INFO, to_console: bool = False) -> logging.Logger:
    """
    Returns a module-scoped logger that writes **only** to
    <log_dir>/<name>.log (plus Rich to the driver if asked).
    Call exactly once per process.
    """
    path = pathlib.Path(log_dir) / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False          # keep noise out of the root logger

    # file (rotating) handler
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    logger.addHandler(fh)

    # optional colourful console for the *driver* only
    if to_console:
        ch = RichHandler(rich_tracebacks=True, markup=True)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    return logger

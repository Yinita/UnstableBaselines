import logging, os, pathlib #, datetime
from rich.logging import RichHandler


# def create_output_folder(run_name):
#     output_dir = os.path.join("outputs", str(datetime.datetime.now().strftime('%Y-%m-%d')), str(datetime.datetime.now().strftime('%H-%M-%S')), run_name)
#     os.makedirs(output_dir)
#     output_dirs = []
#     for folder_name in ["training_data", "eval_data", "checkpoints", "logs"]: 
#         output_dirs.append(os.path.join(self.output_dir, folder_name)); os.makedirs(output_dirs[-1], exist_ok=True)
#     return *output_dirs

def setup_logger(name: str, log_dir: str, level: int = logging.INFO, to_console: bool = False) -> logging.Logger:
    path = pathlib.Path(log_dir) / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # keep noise out of the root logger
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)
    if to_console: # optional colourful console for the *driver* only
        ch = RichHandler(rich_tracebacks=True, markup=True)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger

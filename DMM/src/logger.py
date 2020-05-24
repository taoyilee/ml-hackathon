import logging

default_format = '%(levelname)s \t %(message)s'
logger = logging.getLogger("wildfire")
logger.setLevel(logging.INFO)

if not logging.root.handlers:
    default_handler = logging.StreamHandler()
    default_handler.setLevel(logging.INFO)
    default_handler.setFormatter(logging.Formatter(default_format))
    logger.addHandler(default_handler)
    logger.propagate = False

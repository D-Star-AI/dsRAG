import logging

# Configure the dsparse logger with a NullHandler to prevent "No handler found" warnings
logger = logging.getLogger("dsrag.dsparse")
logger.addHandler(logging.NullHandler())
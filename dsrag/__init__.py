import logging

# Configure the root dsrag logger with a NullHandler to prevent "No handler found" warnings
# This follows Python best practices for library logging
# Users will need to configure their own handlers if they want to see dsrag logs
logger = logging.getLogger("dsrag")
logger.addHandler(logging.NullHandler())
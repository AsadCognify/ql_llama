import logging


# Creating handlers
file_handler = logging.FileHandler("app.log")
console_handler = logging.StreamHandler()

# Set the logging level for each handler explicitly
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)

# Define the format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s [in %(pathname)s:%(lineno)d]')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Get the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Ensure the logger does not propagate to the root logger
logger.propagate = False

# Test logging
logger.info("Logger initialized successfully")
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")
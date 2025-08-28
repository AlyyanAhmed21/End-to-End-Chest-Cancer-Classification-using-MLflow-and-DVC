import sys
import logging

# We will log directly to the console (stdout), which Hugging Face captures.
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.StreamHandler(sys.stdout) # Only log to the console
    ]
)

logger = logging.getLogger("cnnClassifierLogger")
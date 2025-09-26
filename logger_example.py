import logging

logger = logging.getLogger(__name__)

try:
    import flash_attn
except ImportError as e:
    logger.warning(f"The following error occured during the import time {e}")
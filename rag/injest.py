from dotenv import load_dotenv

load_dotenv()

from rag.faq_data import ECOMMERCE_FAQ
from rag.pipeline import ingest_faq
from logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Starting FAQ ingestion")
    ingest_faq(ECOMMERCE_FAQ)
    logger.info("Ingestion complete")

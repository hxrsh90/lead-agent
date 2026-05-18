import asyncio
import time
import schedule
from loguru import logger


def _trigger() -> None:
    logger.info("[Scheduler] Triggering daily pipeline run")
    from main import run_pipeline
    asyncio.run(run_pipeline())


schedule.every().day.at("09:00").do(_trigger)

if __name__ == "__main__":
    logger.info("[Scheduler] Running — pipeline fires daily at 09:00 PT")
    while True:
        schedule.run_pending()
        time.sleep(60)

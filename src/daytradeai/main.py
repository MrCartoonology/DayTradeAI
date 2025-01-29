from typing import Any, Dict
from logging import getLogger, basicConfig, INFO
import daytradeai.data as data
import daytradeai.config as config
import daytradeai.preprocess as preprocess


basicConfig(level=INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)


def main(cfg: Dict[str, Any]) -> None:
    logger.info("Starting main process")
    df_current = data.get_downloaded_data(cfg=cfg["data"])
    df_new = data.get_new_data(cfg=cfg["data"], df_current=df_current)
    data.save_downloaded_data(df=df_new, cfg=cfg["data"])
    df_raw = data.combine_dataframes(df_current, df_new)
    df_preprocessed = preprocess.preprocess_data(
        df=df_raw, data_cfg=cfg["data"], preprocess_cfg=cfg["preprocess"]
    )
    preprocess.save_preprocessed(df=df_preprocessed, cfg=cfg["preprocess"])
    model = train_model()
    evaluate_model(model)
    save_model(model)
    logger.info("Main process completed")


def train_model():
    logger.info("Training model...")
    return "model"


def evaluate_model(model):
    logger.info("Evaluating model...")
    logger.info(model)


def save_model(model):
    logger.info("Saving model...")
    logger.info(model)


if __name__ == "__main__":
    main(cfg=config.cfg)

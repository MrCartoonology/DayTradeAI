from typing import Dict
import pandas as pd
from logging import getLogger, basicConfig, INFO
import daytradeai.data as data
from daytradeai.config import cfg, cfg_dbg
from daytradeai.preprocess import preprocess_data


basicConfig(level=INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = getLogger(__name__)


def main(cfg: Dict[str, str]) -> None:
    logger.info("Starting main process")
    df_current = data.get_downloaded_data(cfg=cfg['data'])
    df_new = data.get_new_data(cfg=cfg['data'], df_current=df_current)
    data.save_downloaded_data(df=df_new, cfg=cfg['data'])
    df_raw = pd.concat([df_current, df_new], axis=1)
    df_preprocessed = preprocess_data(df_raw, cfg=cfg['preprocess'])
    model = train_model()
    evaluate_model(model)
    save_model(model)
    logger.info("Main process completed")
    

def train_model():
    logger.info('Training model...')
    return 'model'


def evaluate_model(model):
    logger.info('Evaluating model...')
    logger.info(model)


def save_model(model):
    logger.info('Saving model...')
    logger.info(model)
    

if __name__ == '__main__':
    main(cfg=cfg)
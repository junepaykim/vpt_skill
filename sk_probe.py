#!/usr/bin/env python3
"""
Setup and load a model, then display its configuration and details.
"""
import os
import torch
import numpy as np
import random
import warnings
from time import sleep
from random import randint

import src.utils.logging as logging
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.models.build_model import build_model
from src.utils.file_io import PathManager
from src.engine.evaluator import Evaluator
from src.engine.trainer import Trainer


from launch import default_argument_parser, logging_train_setup

warnings.filterwarnings("ignore")


def get_loaders(cfg, logger):
    logger.info("Loading training data (final training data for vtab)...")
    train_loader = data_loader.construct_train_loader(cfg)

    logger.info("Loading validation data...")
    # not really needed for vtab
    val_loader = data_loader.construct_val_loader(cfg)
    logger.info("Loading test data...")
    if cfg.DATA.NO_TEST:
        logger.info("...no test data is constructed")
        test_loader = None
    else:
        test_loader = data_loader.construct_test_loader(cfg)
    return train_loader, val_loader, test_loader


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DIST_INIT_PATH = "tcp://{}:12399".format(
        os.environ.get("SLURMD_NODENAME", "localhost")
    )

    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_parameter = f"lr{lr}_wd{wd}"

    output_path = os.path.join(output_dir, output_parameter)
    if not PathManager.exists(output_path):
        PathManager.mkdirs(output_path)

    cfg.OUTPUT_DIR = output_path
    cfg.freeze()
    return cfg


def main(args):
    """
    Main function to set up the configuration and model, then display model details.
    """
    cfg = setup(args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)

    model, cur_device = build_model(cfg)
    evaluator = Evaluator()
    trainer = Trainer(cfg, model, evaluator, cur_device)

    logger = logging.get_logger("visual_prompt")
    prompt_path = cfg.PROMPT_DIR
    train_loader, _, test_loader = get_loaders(cfg, logger)
    trainer.load_prompt(model, prompt_path, train_loader)

    if test_loader:
        trainer.eval_prompt(test_loader, "test_evaluation", save=True)
    else:
        logger.info("No test data available for evaluation.")



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)

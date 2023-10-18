import os

import torch
import torch.distributed as dist

from config import get_config
from runner import MultiWOZRunner

from utils.io_utils import get_or_create_logger

import fitlog
import traceback


def main():
    """ main function """
    cfg = get_config()

    # cuda setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = min(torch.cuda.device_count(), cfg.num_gpus)

    logger = get_or_create_logger(__name__, cfg.model_dir)

    if torch.cuda.is_available():
        if num_gpus > 1:
            cfg.local_rank = int(os.environ["LOCAL_RANK"])  # TODO
            if cfg.local_rank in [0, -1]:
                logger.info('Using Multi-GPU training, number of GPU is {}'.format(num_gpus))
            torch.cuda.set_device(cfg.local_rank)
            device = torch.device('cuda', cfg.local_rank)
            dist.init_process_group(backend='nccl')
        else:
            logger.info('Using single GPU training.')
            device = torch.device('cuda')
        logger.info('CUDA_VISIBLE_DEVICES: ' + str(os.environ["CUDA_VISIBLE_DEVICES"]))
    else:
        device = "cpu"

    setattr(cfg, "device", device)
    setattr(cfg, "num_gpus", num_gpus)

    if cfg.local_rank in [0, -1]:
        logger.info("Device: %s (the number of GPUs: %d)", str(device), num_gpus)

    # if cfg.seed > 0:
    #     random.seed(cfg.seed)
    #     np.random.seed(cfg.seed)
    #     torch.manual_seed(cfg.seed)
    #     torch.cuda.manual_seed_all(cfg.seed)

    fitlog.set_log_dir('logs/')
    rnd_seed = fitlog.set_rng_seed() if cfg.seed == -1 else fitlog.set_rng_seed(cfg.seed)
    cfg.seed = rnd_seed

    fitlog.add_hyper(cfg)
    fitlog.add_hyper_in_file(__file__)

    if cfg.local_rank in [0, -1]:
        logger.info("Set random seed to %d", cfg.seed)

    runner = MultiWOZRunner(cfg)

    try:
        if cfg.run_type == "train":
            runner.train()
        else:
            runner.predict(0)
    except:
        traceback.print_exc()
        fitlog.finish(status=1)

    fitlog.finish()


if __name__ == "__main__":
    main()

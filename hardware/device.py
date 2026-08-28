import logging

import torch


def get_device(force_cpu):
    """
    Select the compute device to run on.
    :param force_cpu: If True, run on CPU even when CUDA is available
    :return: torch.device
    """
    if torch.cuda.is_available() and not force_cpu:
        logging.info("CUDA detected. Running with GPU acceleration.")
        return torch.device("cuda")

    if force_cpu and torch.cuda.is_available():
        logging.info(
            "CUDA detected, but overriding with option '--cpu'. Running with only CPU."
        )
    else:
        logging.info("CUDA is *NOT* detected. Running with only CPU.")

    return torch.device("cpu")

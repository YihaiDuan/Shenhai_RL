"""
@Time ： 2021/2/7 15:26
@Auth ： Duan Yihai
@File ：utils.py
@email ：duanyihai@tju.edu.cn
@Motto：Keep Coding, Keep Thinking
"""
import argparse
import logging
import time
import os

def str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Must be true or false')


def loadLogger(path='info', name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt="[ %(asctime)s  %(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)

    logger.addHandler(sHandler)
    if path != "":
        work_dir = os.path.join(path,
                                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
        fHandler.setLevel(logging.DEBUG)
        fHandler.setFormatter(formatter)

        logger.addHandler(fHandler)

    return logger
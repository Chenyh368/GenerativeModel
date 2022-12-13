from utils.experiman import manager
from options.train_options import TrainOptions

import os


if __name__ == '__main__':
    parser = manager.get_basic_arg_parser()
    opt = TrainOptions(parser).parse()   # get training options
    manager.setup(opt, third_party_tools=('tensorboard',))
    logger = manager.get_logger()
    device = 'cuda'

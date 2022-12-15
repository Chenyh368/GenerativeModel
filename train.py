from utils.experiman import manager
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import time
from utils import util
from utils.visualizer import Visualizer
from tqdm import tqdm
import os


if __name__ == '__main__':
    parser = manager.get_basic_arg_parser()
    opt = TrainOptions(parser).parse()   # get training options
    manager.setup(opt, third_party_tools=('tensorboard',))
    logger = manager.get_logger()
    device = 'cuda'

    logger.info("======> Preparing Data")
    dataset = create_dataset(opt, manager)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training images = %d' % dataset_size)

    logger.info("======> Preparing Model")
    model = create_model(opt, manager)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt, manager)  # create a visualizer that display/save images and plots
    total_iters = 0

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        logger.info(f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}")
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        t = tqdm(dataset, total=dataset.iter_num_per_epoch(), desc=f"Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}",  dynamic_ncols = True)

        for i, data in enumerate(t):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            # if total_iters % opt.print_freq == 0:
            #     pass

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.iter_display_freq == 0:  # display images on visdom and save images to a HTML file
                model.compute_visuals()
                visuals = model.get_current_visuals()
                images = []
                for label, image in visuals.items():
                    manager.log_image(label, (image + 1) / 2, total_iters, epoch, split="Train")

            #
            # if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     pass
            #
            # if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
            #     pass
            if total_iters % opt.iter_print_freq == 0:
                iter_data_time = time.time()
                losses = model.get_current_losses()
                message = ""
                for k, v in losses.items():
                    message += '%s: %.3f |' % (k, v)
                    manager.log_metric(k, v, total_iters, epoch, split="Train")
                t.set_postfix_str(message)


        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logger.info('====> saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        # logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
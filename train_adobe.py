import argparse
import os

import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader, default_collate

from data_loader import AdobeDataAffineHR
from decorators import except_errors
from functions import *
from loss_functions import alpha_gradient_loss, alpha_loss, compose_loss
from networks import ResnetConditionHR, conv_init
from time_ import get_time, print_time_elapsed


def collate_filter_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


@print_time_elapsed
@except_errors()
def main():
    # CUDA

    # os.environ["CUDA_VISIBLE_DEVICES"]="4"
    # print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])
    print(f'Is CUDA available: {torch.cuda.is_available()}')

    """Parses arguments."""
    parser = argparse.ArgumentParser(description = 'Training Background Matting on Adobe Dataset')
    parser.add_argument('-n', '--name', type = str,
                        help = 'Name of tensorboard and model saving folders')
    parser.add_argument('-bs', '--batch_size', type = int,
                        help = 'Batch Size')
    parser.add_argument('-res', '--reso', type = int,
                        help = 'Input image resolution')

    parser.add_argument('-cont', '--continue', action = 'store_true',
                        help = 'Indicates to run the continue training using the latest saved model')
    parser.add_argument('-w', '--workers', type = int, default = None,
                        help = 'Number of worker to load data')
    parser.add_argument('-ep', '--epochs', type = int, default = 60,
                        help = 'Maximum Epoch')
    parser.add_argument('-n_blocks1', '--n_blocks1', type = int, default = 7,
                        help = 'Number of residual blocks after Context Switching')
    parser.add_argument('-n_blocks2', '--n_blocks2', type = int, default = 3,
                        help = 'Number of residual blocks for Fg and alpha each')

    args = parser.parse_args()
    if args.workers is None:
        args.workers = args.batch_size

    continue_training = getattr(args, 'continue')

    # Directories
    tb_dir = f'tb_summary/{args.name}'
    model_dir = f'models/{args.name}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # Input list
    data_config_train = {
        'reso'   : [args.reso, args.reso],
        'trimapK': [5, 5],
        'noise'  : True
    }  # choice for data loading parameters

    # DATA LOADING
    print('\n[Phase 1] : Data Preparation')

    # Original Data
    traindata = AdobeDataAffineHR(
        csv_file = 'Data_adobe/Adobe_train_data.csv',
        data_config = data_config_train,
        transform = None
    )  # Write a dataloader function that can read the database provided by .csv file

    train_loader = DataLoader(
        traindata,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers,
        collate_fn = collate_filter_none
    )

    print('\n[Phase 2] : Initialization')

    # Find latest saved model
    model, optim = '', ''
    start_epoch = 0
    if continue_training:
        for name in os.listdir(model_dir):
            if name.endswith('.pth') and name.startswith('net_epoch_'):
                ep = int(name[len('net_epoch_'):-4])
                if ep > start_epoch:
                    start_epoch = ep
                    model = name
        if model:
            model = f'{model_dir}/{model}'
            optim = f'{model_dir}/optim_epoch_{start_epoch}.pth'
        else:
            continue_training = False

    net = ResnetConditionHR(input_nc = (3, 3, 1, 4), output_nc = 4, n_blocks1 = 7, n_blocks2 = 3,
                            norm_layer = nn.BatchNorm2d)
    net.apply(conv_init)
    net = nn.DataParallel(net)
    if continue_training:
        net.load_state_dict(torch.load(model))
    net.cuda()
    torch.backends.cudnn.benchmark = True

    # Loss
    l1_loss = alpha_loss()
    c_loss = compose_loss()
    g_loss = alpha_gradient_loss()

    optimizer = Adam(net.parameters(), lr = 1e-4)
    if continue_training:
        optimizer.load_state_dict(torch.load(optim))

    log_writer = SummaryWriter(tb_dir)

    print('Starting Training')
    step = 50  # steps to visualize training images in tensorboard

    KK = len(train_loader)

    for epoch in range(start_epoch, args.epochs):

        net.train()

        netL, alL, fgL, fg_cL, al_fg_cL, elapse_run, elapse = 0, 0, 0, 0, 0, 0, 0

        t0 = get_time()
        testL = 0
        ct_tst = 0

        for i, data in enumerate(train_loader):
            # Initiating

            fg = data['fg'].cuda()
            bg = data['bg'].cuda()
            alpha = data['alpha'].cuda()
            image = data['image'].cuda()
            bg_tr = data['bg_tr'].cuda()
            seg = data['seg'].cuda()
            multi_fr = data['multi_fr'].cuda()

            mask = (alpha > -0.99).type(torch.FloatTensor).cuda()
            mask0 = torch.ones(alpha.shape).cuda()

            tr0 = get_time()

            alpha_pred, fg_pred = net(image, bg_tr, seg, multi_fr)

            ## Put needed loss here
            al_loss = l1_loss(alpha, alpha_pred, mask0)
            fg_loss = l1_loss(fg, fg_pred, mask)

            al_mask = (alpha_pred > 0.95).type(torch.FloatTensor).cuda()
            fg_pred_c = image * al_mask + fg_pred * (1 - al_mask)

            fg_c_loss = c_loss(image, alpha_pred, fg_pred_c, bg, mask0)

            al_fg_c_loss = g_loss(alpha, alpha_pred, mask0)

            loss = al_loss + 2 * fg_loss + fg_c_loss + al_fg_c_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            netL += loss.data
            alL += al_loss.data
            fgL += fg_loss.data
            fg_cL += fg_c_loss.data
            al_fg_cL += al_fg_c_loss.data

            log_writer.add_scalar('training_loss', loss.data, epoch * KK + i + 1)
            log_writer.add_scalar('alpha_loss', al_loss.data, epoch * KK + i + 1)
            log_writer.add_scalar('fg_loss', fg_loss.data, epoch * KK + i + 1)
            log_writer.add_scalar('comp_loss', fg_c_loss.data, epoch * KK + i + 1)
            log_writer.add_scalar('alpha_gradient_loss', al_fg_c_loss.data, epoch * KK + i + 1)

            t1 = get_time()

            elapse += t1 - t0
            elapse_run += t1 - tr0

            t0 = t1

            testL += loss.data
            ct_tst += 1

            if i % step == (step - 1):
                print(
                    f'[{epoch + 1}, {i + 1:5d}] '
                    f'Total-loss: {netL / step:.4f} '
                    f'Alpha-loss: {alL / step:.4f} '
                    f'Fg-loss: {fgL / step:.4f} '
                    f'Comp-loss: {fg_cL / step:.4f} '
                    f'Alpha-gradient-loss: {al_fg_cL / step:.4f} '
                    f'Time-all: {elapse / step:.4f} '
                    f'Time-fwbw: {elapse_run / step:.4f}'
                )
                netL, alL, fgL, fg_cL, al_fg_cL, elapse_run, elapse = 0, 0, 0, 0, 0, 0, 0

                write_tb_log(image, 'image', log_writer, i)
                write_tb_log(seg, 'seg', log_writer, i)
                write_tb_log(alpha, 'alpha', log_writer, i)
                write_tb_log(alpha_pred, 'alpha_pred', log_writer, i)
                write_tb_log(fg * mask, 'fg', log_writer, i)
                write_tb_log(fg_pred * mask, 'fg_pred', log_writer, i)
                write_tb_log(multi_fr[0:4, 0, ...].unsqueeze(1), 'multi_fr', log_writer, i)

                # composition
                alpha_pred = (alpha_pred + 1) / 2
                comp = fg_pred * alpha_pred + (1 - alpha_pred) * bg
                write_tb_log(comp, 'composite', log_writer, i)
                del comp

            del fg, bg, alpha, image, alpha_pred, fg_pred, bg_tr, seg, multi_fr

        # Saving
        torch.save(net.state_dict(), f'{model_dir}/net_epoch_{epoch + 1}.pth')
        torch.save(optimizer.state_dict(), f'{model_dir}/optim_epoch_{epoch + 1}.pth')


if __name__ == '__main__':
    main()

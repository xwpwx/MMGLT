import argparse
import torch

from torch.autograd import Variable
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
from models.model import make_model
from datasets.make_dataloader import make_dataset
from tools.utils_server import save_network, copyfiles2checkpoints, get_logger
from tools.evaltools import evaluate
import warnings
from losses.balanceLoss import LossFunc
from tqdm import tqdm
import numpy as np
import cv2
import json
from tools.JWD_M import Distance
import os
from torch import optim
from tools.flops import get_model_complexity_info
import beepy
import random



warnings.filterwarnings("ignore")
map_all = [3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5]
save_metre_rnage = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
save_metre_original = (np.array([0] * (len(save_metre_rnage) + 1) * len(map_all)).reshape(len(map_all),
                                                                                          len(save_metre_rnage) + 1)).tolist()
save_metre_xy = (np.array([0] * (len(save_metre_rnage) + 1) * len(map_all)).reshape(len(map_all),
                                                                                    len(save_metre_rnage) + 1)).tolist()
d_m_all = 0
save_metre_original_all = save_metre_original[0].copy()
save_metre_xy_all = save_metre_xy[0].copy()


def create_hanning_mask(center_R):
    hann_window = np.outer(
        np.hanning(center_R + 2),
        np.hanning(center_R + 2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1, 1:-1]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default="0", type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--las_mode1',   default="rgb", type=str,help='false,gray,hillshade,slope')
    parser.add_argument('--las_mode2',   default="hillshade", type=str,help='false,gray,hillshade,slope')
    parser.add_argument('--train_part_whole', default="5x", type=str, help='part or whole')
    parser.add_argument('--val_part_whole', default="whole", type=str, help='part or whole')
    parser.add_argument('--name', default="test", type=str, help='file name')
    parser.add_argument('--backbone', default="MMGLT", type=str,
                        help='sbtv2_base')
    parser.add_argument('--neck', default="global_up", type=str, help='neck_name')
    parser.add_argument('--data_dir', default='/home/xwp/windows_d/dataset/tile_500',
                        type=str, help='training dir path')
    parser.add_argument('--centerR', default=33, type=int, help='')
    parser.add_argument('--UAVhw', default=96, type=int, help='')
    parser.add_argument('--Satellitehw', default=256, type=int, help='')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate four 0 for sbt, five 0 for mamba')
    parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
    parser.add_argument('--neg_weight', default=15.0, type=float, help='balance sample')
    parser.add_argument('--num_epochs', default= 10, type=int, help='whole epoch ')
    parser.add_argument('--start_save', default= 10, type=int, help='start save')
    parser.add_argument('--start_test', default= 10, type=int, help='start test')
    parser.add_argument('--save_epochs', default=10 , type=int, help='each epochs save')
    parser.add_argument('--save_ckpt', default=True, type=float, help='save pth or not')
    parser.add_argument('--num_worker', default=6, type=int, help='')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--autocast', action='store_true', default=True, help='use mix precision')
    parser.add_argument('--log_iter', default=100, type=int, help='')
    parser.add_argument('--padding', default=0, type=float, help='the times of padding for the image size')
    parser.add_argument('--share', default=False, type=int, help='the times of padding for the image size')
    parser.add_argument('--steps', default=[6, 8, 12, 18], type=list,
                        help='the step of lr_scheduler')
    parser.add_argument('--old_model_pretrain', default='/home/xwp/work/three_sspt/checkpoints/1.3_rxh_doub_0/net_060.pth', help='all model will be loaded if you use this way')
    parser.add_argument('--pos_num', default=300, type=int, help='the times of padding for the image size')
    parser.add_argument('--loc_loss', default="smooth_l1_loss", type=str, help='MSE BCE smooth_l1_loss')
    parser.add_argument('--loc_weight', default=1.0, type=float, help='BCE smooth_l1_loss')
    parser.add_argument('--loc_label', default="400", type=str, help='400 or 1')
    parser.add_argument('--pos_label', default=33, type=int, help='400 or 1')
    parser.add_argument('--adamw_cos', default=5e-6, type=float, help='400 or 1')
    parser.add_argument('--cut_circle', default=1500, type=int, help='400 or 1')
    parser.add_argument('--cover_rate', default=0.85, type=float, help='400 or 1')
    parser.add_argument('--NEK_W', default=1.5, type=float, help='400 or 1')
    opt = parser.parse_args()
    opt.UAVhw = [opt.UAVhw, opt.UAVhw]
    opt.Satellitehw = [opt.Satellitehw, opt.Satellitehw]
    return opt


def evaluate_distance(X, Y, opt, sa_path, bias=False):
    global map_all
    global save_metre_original
    global save_metre_xy
    global save_metre_rnage
    global d_m_all

    get_gps_x = X / opt.Satellitehw[0]
    get_gps_y = Y / opt.Satellitehw[0]
    path = sa_path[0].split("/")
    read_gps = json.load(
        open(sa_path[0].split("/Satellite")[0] + "/GPS_info.json", 'r', encoding="utf-8"))
    tl_E = read_gps["Satellite"][path[-1]]["tl_E"]
    tl_N = read_gps["Satellite"][path[-1]]["tl_N"]
    br_E = read_gps["Satellite"][path[-1]]["br_E"]
    br_N = read_gps["Satellite"][path[-1]]["br_N"]
    map_size = read_gps["Satellite"][path[-1]]["map_size"]
    UAV_GPS_E = read_gps["LAS_mid"]["lon"]
    UAV_GPS_N = read_gps["LAS_mid"]["lat"]
    PRE_GPS_E = tl_E + (br_E - tl_E) * get_gps_y
    PRE_GPS_N = tl_N - (tl_N - br_N) * get_gps_x 

    d_m = Distance(UAV_GPS_N, UAV_GPS_E, PRE_GPS_N, PRE_GPS_E)
    map_index = map_all.index(map_size)

    if bias == False:
        save_metre_original[map_index][21] = save_metre_original[map_index][21] + 1 
        save_metre_original_all[21] = save_metre_original_all[21] + 1 
        for i in range(len(save_metre_rnage)):
            if d_m <= save_metre_rnage[i]:
                save_metre_original_all[i] = save_metre_original_all[i] + 1
                save_metre_original[map_index][i] = save_metre_original[map_index][i] + 1
    else:
        save_metre_xy[map_index][21] = save_metre_xy[map_index][21] + 1
        save_metre_xy_all[21] = save_metre_xy_all[21] + 1
        for i in range(len(save_metre_rnage)):
            if d_m <= save_metre_rnage[i]:
                save_metre_xy_all[i] = save_metre_xy_all[i] + 1
                save_metre_xy[map_index][i] = save_metre_xy[map_index][i] + 1
    return 0


def train_model(model, opt, dataloaders, dataset_sizes):
    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs

    file_name = f"{opt.name}_{0}" 
    dir_name = './checkpoints'

    counter = 1
    if os.path.exists(os.path.join(dir_name, file_name)):
        while os.path.exists(os.path.join(dir_name, f"{opt.name}_{counter}")):
            counter += 1
        file_name = f"{opt.name}_{counter - 1}"
    else:
        file_name =  file_name
    new_file_path = os.path.join(dir_name, file_name)
    logger = get_logger(new_file_path + "/train.log")

    since = time.time()
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch
    scaler = GradScaler()
    criterion = LossFunc(opt.centerR, opt.neg_weight, opt.gpu_ids,opt=opt)
    logger.info('start training!')
    logger.info("GFLOPs: {}".format(flop))
    logger.info("Params: {}".format(param))

    optimizer, scheduler = make_optimizer(model, opt)

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        logger.info('-' * 30)

        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        iter_cls_loss = 0.0
        iter_loc_loss = 0.0
        iter_start = time.time()
        iter_loss = 0

        # train
        for iter, (z, y, x, ratex, ratey) in enumerate(dataloaders["train"]):
            now_batch_size, _, _, _ = z.shape
            total_iters = len(dataloaders["train"])
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            if use_gpu:
                z = Variable(z.cuda().detach())
                y = Variable(y.cuda().detach())
                x = Variable(x.cuda().detach())
            else:
                z, y, x = Variable(z), Variable(y), Variable(x)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(z,y,x)  # satellite and drone
            cls_loss, loc_loss = criterion(outputs, [ratex, ratey])
            loc_loss = loc_loss
            loss = cls_loss + loc_loss
            # backward + optimize only if in training phase
            if epoch < opt.warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss_backward = warm_up * loss
            else:
                loss_backward = loss

            if opt.autocast:
                scaler.scale(loss_backward).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_backward.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * now_batch_size
            iter_loss += loss.item() * now_batch_size
            iter_cls_loss += cls_loss.item() * now_batch_size
            iter_loc_loss += loc_loss.item() * now_batch_size

            if (iter + 1) % opt.log_iter == 0:
                time_elapsed_part = time.time() - iter_start
                iter_loss = iter_loss / opt.log_iter / now_batch_size
                iter_cls_loss = iter_cls_loss / opt.log_iter / now_batch_size
                iter_loc_loss = iter_loc_loss / opt.log_iter / now_batch_size

                lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

                logger.info("[{}/{}] loss: {:.6f} cls_loss: {:.6f} loc_loss:{:.4f} lr_backbone:{:.6f}"
                            "time:{:.0f}m {:.0f}s ".format(iter + 1,
                                                           total_iters,
                                                           iter_loss,
                                                           iter_cls_loss,
                                                           iter_loc_loss,
                                                           lr_backbone,
                                                           time_elapsed_part // 60,
                                                           time_elapsed_part % 60))
                iter_loss = 0.0
                iter_loc_loss = 0.0
                iter_cls_loss = 0.0
                iter_start = time.time()

        epoch_loss = running_loss / dataset_sizes['satellite']

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

        time_elapsed = time.time() - since
        logger.info('Epoch[{}/{}] Loss: {:.6f}  lr_backbone:{:.6f} time:{:.0f}m {:.0f}s'.format(epoch + 1,
                                                                                                num_epochs,
                                                                                                epoch_loss,
                                                                                                lr_backbone,
                                                                                                time_elapsed // 60,
                                                                                                time_elapsed % 60))
        # deep copy the model
        scheduler.step()
        if (epoch + 1) >= opt.start_save and (epoch + 1) % opt.save_epochs == 0:
            if opt.save_ckpt:
                save_network(model, file_name, epoch + 1)

            model.eval()
        if (epoch + 1) >= opt.start_test and (epoch + 1) % opt.save_epochs == 0:
            total_score = 0.0
            total_score_b = 0.0
            start_time = time.time()
            flag_bias = 0
            for uav1,uav2, satellite, X, Y, uav_path1,uav_path2, sa_path in tqdm(dataloaders["val"]):
                z = uav1.cuda()
                y = uav2.cuda()
                x = satellite.cuda()

                response, loc_bias = model(z, y, x)
                response = torch.sigmoid(response)
                map = response.squeeze().cpu().detach().numpy()

                if opt.centerR != 1:
                    kernel = create_hanning_mask(opt.centerR)
                    map = cv2.filter2D(map, -1, kernel)

                label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])

                satellite_map = cv2.resize(map, opt.Satellitehw)
                id = np.argmax(satellite_map)
                S_X = int(id // opt.Satellitehw[0])
                S_Y = int(id % opt.Satellitehw[1])

                pred_XY = np.array([S_X, S_Y])
                single_score = evaluate(opt, pred_XY=pred_XY, label_XY=label_XY)
                total_score += single_score
                evaluate_distance(S_X, S_Y, opt, sa_path, bias=False)  #

                if loc_bias is not None:
                    flag_bias = 1
                    loc = loc_bias.squeeze().cpu().detach().numpy()
                    id_map = np.argmax(map)
                    S_X_map = int(id_map // map.shape[-1])
                    S_Y_map = int(id_map % map.shape[-1])
                    pred_XY_map = np.array([S_X_map, S_Y_map])
                    if opt.loc_label=="400":
                        # pred_XY_b = (pred_XY_map + (loc[:, S_X_map, S_Y_map]*51-25.5)) * opt.Satellitehw[0] / loc.shape[-1]  # add bias
                        pred_XY_b = (pred_XY_map + (loc[:, S_X_map, S_Y_map] )) * opt.Satellitehw[0] / \
                                    loc.shape[-1]  # add bias
                    else:
                        pred_XY_b = (pred_XY_map + loc[:, S_X_map, S_Y_map]*opt.Satellitehw[0]) * opt.Satellitehw[0] / loc.shape[-1]  # add bias

                    pred_XY_b = np.array(pred_XY_b)
                    single_score_b = evaluate(opt, pred_XY=pred_XY_b, label_XY=label_XY)
                    total_score_b += single_score_b
                    evaluate_distance(pred_XY_b[0], pred_XY_b[1], opt, sa_path, bias=True)  #

            logger.info('original:3m:{}  5m:{}  10m:{}  20m:{}  30m:{}  40m:{}  50m:{}'.format(
                save_metre_original_all[0] / save_metre_original_all[21],
                save_metre_original_all[1] / save_metre_original_all[21],
                save_metre_original_all[2] / save_metre_original_all[21],
                save_metre_original_all[4] / save_metre_original_all[21],
                save_metre_original_all[6] / save_metre_original_all[21],
                save_metre_original_all[8] / save_metre_original_all[21],
                save_metre_original_all[10] / save_metre_original_all[21]))
            if loc_bias is not None:
                logger.info('bias   :3m:{}  5m:{}  10m:{}  20m:{}  30m:{}  40m:{}  50m:{}'.format(
                    save_metre_xy_all[0] / save_metre_xy_all[21],
                    save_metre_xy_all[1] / save_metre_xy_all[21],
                    save_metre_xy_all[2] / save_metre_xy_all[21],
                    save_metre_xy_all[4] / save_metre_xy_all[21],
                    save_metre_xy_all[6] / save_metre_xy_all[21],
                    save_metre_xy_all[8] / save_metre_xy_all[21],
                    save_metre_xy_all[10] / save_metre_xy_all[21]))

            print("pred: " + str(pred_XY) + " label: " + str(label_XY) + " score:{}".format(single_score))

            time_consume = time.time() - start_time
            logger.info("time consume is {}".format(time_consume))

            score = total_score / len(dataloaders["val"])
            logger.info("the final score is {}".format(score))

            if flag_bias:
                score_b = total_score_b / len(dataloaders["val"])
                logger.info("the final score_bias is {}".format(score_b))


if __name__ == '__main__':
    device = 'cuda'
    opt = get_parse()

    str_ids = "0"  # deal gpu number
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu

    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    setup_seed(int.from_bytes(os.urandom(4), byteorder="big"))
    dataloaders_train, dataset_sizes = make_dataset(opt)
    dataloaders_val = make_dataset(opt, train=False)
    dataloaders = {"train": dataloaders_train,
                   "val": dataloaders_val}

    model = make_model(opt, pretrain=True).to(device)
    flop, param = get_model_complexity_info(model, (3, opt.UAVhw[0], opt.UAVhw[1]),(3, opt.UAVhw[0], opt.UAVhw[1]),
                                            (3, opt.Satellitehw[0], opt.Satellitehw[1]), as_strings=True,
                                            print_per_layer_stat=False)

    model = model.cuda(device=gpu_ids[0])

    copyfiles2checkpoints(opt, path=os.path.basename(__file__))

    train_model(model, opt, dataloaders, dataset_sizes)

    beepy.beep(sound=5)

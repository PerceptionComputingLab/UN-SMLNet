import os
import sys
from scipy.ndimage.measurements import label
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from utils.losses import weighted_ce_loss, focal_loss
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from model.UN_SMLNet import VNetMultiHead
from dataloaders.la_heart import LAHeart, RandomRotation, CenterCrop, RandomCrop, RandomRotFlip, RandomFlip, ToTensor,RandomGammaCorrection
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA_Seg/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='3DUNSMLNet', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
parser.add_argument('--ratio', type=int, default=4, help='ratio for attention block')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--uncertainty', type=int,  default=True, help='whether use uncertainty guide')
parser.add_argument('--js_weight', type=int,  default=100, help='the weight of js')
parser.add_argument('--attention', type=int,  default=True, help='whether use attention block')
parser.add_argument('--seed', type=int,  default=2019, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../unsmlnet_model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
has_att = args.attention
ratio = args.ratio
js_weight = args.js_weight
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

patch_size = (256, 256, 80)
num_classes = 2

def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
def cross_entropy_map(score, target):
    score_log_soft = F.log_softmax(score,dim=1)
    target = target.float()
    ce_loss = 0
    for i in range(target.shape[1]):
        cross_map_i = target[target==i]=1 * score_log_soft[:,i,:,:,:]
        ce_loss -= cross_map_i.mean()
    print('自己算的：%',ce_loss)
    ce_loss_pytorch = F.cross_entropy(score,target)
    print('pytorch算的：%', ce_loss_pytorch)
    #return cross_map


def un_cross_entropy_loss(js, score, target, js_weight):
    e = 2.71828
    cross_map = F.cross_entropy(score, target, reduction='none')  # cross_entropy_map(score, target)
    # js_ce = torch.exp(js_weight *js) * cross_map #+ cross_map
    js_ce = torch.log(e+js_weight*js) * cross_map  # log_e
    return js_ce.mean()

def ungt_cross_entropy_loss(score, target, js_weight):
    e = 2.71828
    cross_map = F.cross_entropy(score, target, reduction='none')  # cross_entropy_map(score, target)
    # js_ce = torch.exp(js_weight *js) * cross_map #+ cross_map
    js_map, js_mean = jsgt_divergency(score, target)
    js_ce = torch.log(e+js_weight*js_map) * cross_map  # log_e
    loss = js_ce.mean() + js_weight * js_mean
    return loss
def jsgt_divergency(score, target):
    smooth = 1e-10
    s_soft = F.softmax(score)
    t_soft = target
    m_soft = (s_soft+t_soft)/2 + smooth
    sm_kl = torch.sum(s_soft * torch.log(s_soft/m_soft), dim=1)# F.kl_div(s_soft,m_soft.log())
    tm_kl = torch.sum(t_soft * torch.log(t_soft/m_soft), dim=1)
    js = (sm_kl + tm_kl)/2
    return js, js.mean()
def js_divergency(score, target):
    smooth = 1e-10
    s_soft = F.softmax(score)
    t_soft = F.softmax(target)
    m_soft = (s_soft+t_soft)/2 + smooth
    sm_kl = torch.sum(s_soft * torch.log(s_soft/m_soft), dim=1)# F.kl_div(s_soft,m_soft.log())
    tm_kl = torch.sum(t_soft * torch.log(t_soft/m_soft), dim=1)
    js = (sm_kl + tm_kl)/2
    return js, js.mean()

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    #shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    net = VNetMultiHead(n_channels=1, n_classes=num_classes, n_filters=16, normalization='groupnorm',
                        ratio=ratio, has_att=has_att, has_dropout=True)
    net = net.cuda()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',
                       transform=transforms.Compose([
                           RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_val = LAHeart(base_dir=train_data_path,
                     split='val',
                     transform=transforms.Compose([
                         CenterCrop(patch_size),
                         ToTensor()
                     ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,  num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                           worker_init_fn=worker_init_fn)
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    

    writer = SummaryWriter(snapshot_path+'/log',  flush_secs=2)
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    net.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        #lr_ = poly_lr(epoch_num, max_epoch,base_lr,exponent=0.9)
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr_
        for i_batch, sampled_batch in enumerate(trainloader):
            # generate paired iput
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            decoder_output, encoder_output = net(volume_batch)
            
            with torch.no_grad():
                js_map, js_mean = js_divergency(decoder_output, encoder_output)
            # compute CE + Dice loss
            loss_ce_de = F.cross_entropy(decoder_output, label_batch)# focal_loss(decoder_output,label_batch)#

            outputs_soft_de = F.softmax(decoder_output, dim=1)
            seg_de = outputs_soft_de.data.max(1)[1]  # .cpu().numpy()
            loss_dice_de = dice_loss(outputs_soft_de[:, 1, :, :, :], label_batch == 1)
            if args.uncertainty:
                # prediction之间计算js
                js_ce_de = un_cross_entropy_loss(js_map, decoder_output, label_batch, js_weight)
                loss_js_ce_de = js_ce_de + js_weight*js_mean
                #与ground truth计算js
                # js_ce_de = ungt_cross_entropy_loss(decoder_output, label_batch, js_weight)
                # loss_js_ce_de = js_ce_de
                # loss_js_ce_de = torch.exp(js)*loss_ce_de + js
                loss_de = 0.5*(loss_js_ce_de + loss_dice_de)
                #loss_de = loss_js_ce_de
            else:
                loss_de = 0.5*(loss_ce_de + loss_dice_de)
                # loss_de = loss_ce_de
            # compute L1 Loss
            # loss_dist = torch.norm(out_dis-gt_dis, 1)/torch.numel(out_dis)
            # compute encoder CE + Dice loss
            loss_ce_en = F.cross_entropy(encoder_output, label_batch)# focal_loss(encoder_output,label_batch)#

            outputs_soft_en = F.softmax(encoder_output, dim=1)
            seg_en = outputs_soft_en.data.max(1)[1]  # .cpu().numpy()
            loss_dice_en = dice_loss(outputs_soft_en[:, 1, :, :, :], label_batch == 1)
            if args.uncertainty:
                # prediction之间计算js
                js_ce_en = un_cross_entropy_loss(js_map, encoder_output, label_batch, js_weight)
                loss_js_ce_en = js_ce_en + js_weight*js_mean
                # 与ground truth计算js
                # js_ce_en = ungt_cross_entropy_loss(encoder_output, label_batch, js_weight)
                #loss_js_ce_en = js_ce_en
                # loss_js_ce_en = torch.exp(js) * loss_ce_en + js
                loss_en = 0.5*(loss_js_ce_en + loss_dice_en)
                #loss_en = loss_js_ce_en
            else:
                loss_en = 0.5*(loss_ce_en + loss_dice_en)
                #loss_en = loss_ce_en

            loss = loss_de + loss_en

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_ce_de', loss_ce_de, iter_num)
            writer.add_scalar('loss/loss_dice_de', loss_dice_de, iter_num)
            writer.add_scalar('loss/js', js_mean, iter_num)
            if args.uncertainty:
                writer.add_scalar('loss/js_ce_de', js_ce_de, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : js : %f' % (iter_num, js_mean.item()))
            logging.info('iteration %d : loss_dice_de : %f' % (iter_num, loss_dice_de.item()))
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 2 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft_de[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/de_Predicted_label', grid_image, iter_num)
                image = seg_de[0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/de_Predicted_label_binary', grid_image, iter_num)

                image = outputs_soft_en[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/en_Predicted_label', grid_image, iter_num)
                image = seg_en[0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/en_Predicted_label_binary', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

            ## change lr
            
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            
            
            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            
            if iter_num > max_iterations:
                break
            time1 = time.time()
        
        with torch.no_grad():
            net.eval()
            dice_val = 0.
            val_num = 0.
            for i_val_batch, sampled_val_batch in enumerate(valloader):
                volume_val_batch, label_val_batch = sampled_val_batch['image'], sampled_val_batch['label']
                volume_val_batch, label_val_batch = volume_val_batch.cuda(), label_val_batch.cuda()
                outputs_val_de, outputs_val_en = net(volume_val_batch)
                loss_val_seg = F.cross_entropy(outputs_val_de, label_val_batch)
                outputs_val_soft = F.softmax(outputs_val_de, dim=1)
                val_seg = outputs_val_soft.data.max(1)[1]  # .cpu().numpy()
                inter = torch.sum(val_seg * label_val_batch)
                seg_sum = torch.sum(val_seg)
                lb_sum = torch.sum(label_val_batch)
                dice_val += (2 * inter + 1e-5) / (seg_sum + lb_sum + 1e-5)
                val_num += 1
            dice_val /= val_num
            writer.add_scalar('val/dice', dice_val, epoch_num)
            writer.add_scalar('val/loss_ce', loss_val_seg, epoch_num)
            # view the last val prediction
            image = volume_val_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            grid_image = make_grid(image, 5, normalize=True)
            writer.add_image('val/Image', grid_image, epoch_num)

            image = outputs_val_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            grid_image = make_grid(image, 5, normalize=False)
            writer.add_image('val/Predicted_label', grid_image, epoch_num)

            image = val_seg[0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            grid_image = make_grid(image, 5, normalize=False)
            writer.add_image('val/Predicted_label_binary', grid_image, epoch_num)

            image = label_val_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            grid_image = make_grid(image, 5, normalize=False)
            writer.add_image('val/Groundtruth_label', grid_image, epoch_num)
        if iter_num > max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations+1)+'.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

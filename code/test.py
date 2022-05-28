import argparse
from re import L
import torch
from model.UN_SMLNet import VNetMultiHead

import h5py
import math
import nibabel as nib
import numpy as np
import nrrd
from utils.measures import dc,jc, hd95,assd,hd
from surface_distance import metrics as surf_metric
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import os
import pandas as pd
from matplotlib import pyplot as plt
from collections import OrderedDict
import sys
from utils.util import plot_seg2img
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from skimage.measure import label
###########################
# test UNSMLNet: model: 3DUNSMLNet, test_ml: False
# test SMLNet: model: 3DSMLNet, test_ml: False
# test MLNet: model: 3DMLNet, test_ml: True
# save prediction: save_result:True; save uncertainty map: save_un:True
# do post process: post: True
###########################
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/LA_Seg/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='3DUNSMLNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
parser.add_argument('--ratio', type=int, default=4, help='ratio for attention block')
parser.add_argument('--normalization', type=str,  default='groupnorm', help='normalization to use')
parser.add_argument('--attention', type=int,  default=True, help='whether use attention block')
parser.add_argument('--epoch_num', type=int,  default='6000', help='checkpoint to use')
parser.add_argument('--spacing', type=float,  default=0.625, help='voxel space')
parser.add_argument('--format', type=str, default='h5', help='data format, h5 or nrrd')
parser.add_argument('--post', type=int, default=False, help='whether use cca')
#########
# test model
#########
parser.add_argument('--test_ml', type=int, default=False, help='test MLNet')
#########
# save options
#########
parser.add_argument('--save_un', type=int, default=False, help='whether save uncertainty map')
parser.add_argument('--save_slices', type=int, default=False, help='whether save predictions on slice-level')
parser.add_argument('--save_result', type=int, default=True, help='whether save prediction on volume-level')
parser.add_argument('--save_gt_slices', type=int, default=False, help='whether save ground truth on slice-level')
#########
# test options
#########
parser.add_argument('--step_xy', type=int, default=1, help='sliding window step in xy')
parser.add_argument('--step_z', type=int, default=4, help='sliding window step in z')

FLAGS = parser.parse_args()

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# use cpu to inference
# device = torch.device('cpu')
# use GPU to inference
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../unsmlnet_model/"+FLAGS.model+"/"
test_save_path = "../unsmlnet_model/prediction/"+FLAGS.model+"/"#+"3"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

patch_shape=(256, 256, 80)
num_classes = 2
output_size = (256,256,88)

with open( '../data/test.list', 'r') as f:
    image_list = f.readlines()

if FLAGS.format == 'h5':
    image_list = [FLAGS.root_path + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
elif FLAGS.format == 'nrrd':
    data_path = '../data/3D/Testing Set/'
    image_list = [data_path + item.replace('\n', '') + "/lgemri.nrrd" for item in image_list]

# print(image_list)

def test_calculate_metric(epoch_num):
    if FLAGS.test_ml:
        has_enout = False
    else:
        has_enout = True
    
    net = VNetMultiHead(n_channels=1, n_classes=num_classes, normalization=FLAGS.normalization,
                        ratio=FLAGS.ratio, has_att=FLAGS.attention, has_enout=has_enout, has_dropout=False).cuda()# to(device) # 

    
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()


    avg_metric = dist_test_all_case(net, image_list, num_classes=num_classes,
                                        save_result=FLAGS.save_result, test_save_path=test_save_path,
                                        has_post=FLAGS.post, preproc_fn=True, save_un=FLAGS.save_un)


    return avg_metric


def dist_test_all_case(net, image_list, num_classes, save_result=True, test_save_path=None, preproc_fn=True,has_post=False,save_un=False):
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict['name'] = list()
    metric_dict['dice'] = list()
    metric_dict['jaccard'] = list()
    metric_dict['asd'] = list()
    metric_dict['95hd'] = list()
    metric_dict['hd'] = list()
    metric_dict['nsd'] = list()
    for image_path in tqdm(image_list):
        case_name = image_path.split('/')[-2]
        id = image_path.split('/')[-1]
        '''
        image, _ = nrrd.read(image_path)
        label, _ = nrrd.read(image_path.replace('lgemri', 'laendo'))
        image.astype(np.float32)
        label = (label == 255).astype(np.uint8)
        '''
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        
        if preproc_fn:
            image_map, label_map, minx, maxx, miny, maxy = preprocess_test_data(image,label, output_size)
        else:
            image_map = (image-np.mean(image))/np.std(image) #preproc_fn(image) 
        with torch.no_grad():
            prediction = np.zeros_like(label)
            if save_un:
                prediction_map, score_map,un_map = test_single_case_patch(net, image_map, FLAGS.step_xy, FLAGS.step_z, patch_shape , num_classes=num_classes, save_un=True)
            else:
                # prediction_map, score_map = test_single_case_patch(net, image_map, 1, 4, patch_shape , num_classes=num_classes)
                prediction_map, score_map = test_single_case_patch(net, image_map, FLAGS.step_xy, FLAGS.step_z, patch_shape , num_classes=num_classes)

        if save_result:
            test_save_path_temp = os.path.join(test_save_path, case_name)
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            nib.save(nib.Nifti1Image(prediction_map.astype(np.float32), np.eye(4)), test_save_path_temp + '/' + id + "_pred.nii.gz")
            # nib.save(nib.Nifti1Image(image[minx:maxx,miny:maxy,:].astype(np.float32), np.eye(4)), test_save_path_temp + '/' +  id + "_img.nii.gz")
            # nib.save(nib.Nifti1Image(label[minx:maxx,miny:maxy,:].astype(np.float32), np.eye(4)), test_save_path_temp + '/' + id + "_gt.nii.gz")
            # nib.save(nib.Nifti1Image(score_map[1,...].astype(np.float32), np.eye(4)), test_save_path_temp + '/' + id + "_score.nii.gz")
        if save_un:
            test_save_un_path = test_save_path+'/uncertainty/'+case_name
            if not os.path.exists(test_save_un_path):
                os.makedirs(test_save_un_path)
            un_map = np.transpose(un_map, (2, 1, 0))
            for i in range(88):
                un_slice_name = case_name+'_'+str(i)+'.png'
                un_slice_save_path = os.path.join(test_save_un_path,un_slice_name)
                un_slice = un_map[i,:,:]
                plt.figure()
                plt.imshow(un_slice,cmap=plt.cm.rainbow,interpolation='nearest')
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.savefig(un_slice_save_path)
                plt.clf()
                plt.cla()
                plt.close()
        if FLAGS.save_slices:
            test_save_slice_path = test_save_path+'/slices_with_pred/'+case_name
            if not os.path.exists(test_save_slice_path):
                os.makedirs(test_save_slice_path)
                # (pred_volume,img_volume,gt_volume, save_path, orientation='axial', dice_list=None, isvalue=False,pred_cr='blue', gt_cr='red')
            if FLAGS.format == 'nrrd':
                # plot_seg2img(prediction_map, image[minx:maxx, miny:maxy,:], test_save_slice_path, orientation='coronal')
                # 保存纯图片
                zero_map = np.zeros_like(image_map)
                plot_seg2img(zero_map, image[minx:maxx, miny:maxy,:], test_save_slice_path, orientation='coronal')
            else:
                
                # plot_seg2img(prediction_map, image_map, test_save_slice_path, orientation='coronal')
                # 保存纯图片
                zero_map = np.zeros_like(image_map)
                plot_seg2img(zero_map, image_map, test_save_slice_path, orientation='coronal')
        if FLAGS.save_gt_slices:
            gt_slice_save_path = test_save_path+'/'+case_name+'/slices_with_gt'
            if not os.path.exists(gt_slice_save_path):
                os.makedirs(gt_slice_save_path)
            plot_seg2img(label[minx:maxx, miny:maxy,:], image[minx:maxx, miny:maxy,:], gt_slice_save_path, orientation='coronal',pred_cr='green')

        if FLAGS.format=='nrrd' and preproc_fn:
            prediction[minx:maxx, miny:maxy,:] = prediction_map
        else:
            prediction = prediction_map
        if np.sum(prediction) == 0:
            single_metric = (0,0,0,0,0,0)
        else:
            if has_post:
                print('post')
                prediction = getLargestCC(prediction)
            single_metric = calculate_metric_percase(prediction, label[:], space=FLAGS.spacing)
            metric_dict['name'].append(case_name)
            metric_dict['dice'].append(single_metric[0])
            metric_dict['jaccard'].append(single_metric[1])
            metric_dict['asd'].append(single_metric[2])
            metric_dict['95hd'].append(single_metric[3])
            metric_dict['hd'].append(single_metric[4])
            metric_dict['nsd'].append(single_metric[5])
            print(case_name,single_metric)


        total_metric += np.asarray(single_metric)

        

    avg_metric = total_metric / len(image_list)
    metric_csv = pd.DataFrame(metric_dict)
    if has_post:
        metric_csv.to_csv(test_save_path + '/metric_post_' + str(FLAGS.epoch_num) + '.csv', index=False)
    else:
        metric_csv.to_csv(test_save_path + '/metric_'+str(FLAGS.epoch_num)+'.csv', index=False)
    print('average metric is {}'.format(avg_metric))

    return avg_metric

def test_single_case(net, image, num_classes=1):
    image = np.expand_dims(np.expand_dims(image, axis=0), axis=0).astype(np.float32)
    image = torch.from_numpy(image).cuda()

    y_decoder, y_encoder = net(image)
    # print(y1.shape, out_dist.shape) # ([1, 2, 112, 112, 80]) ([1, 1, 112, 112, 80])
    y_de_soft = F.softmax(y_decoder, dim=1)
    y_en_soft = F.softmax(y_encoder, dim=1)
    y = torch.mean(torch.stack([y_de_soft, y_en_soft]), dim=0)
    y = y.cpu().data.numpy()
    y = y[0, :, :, :, :]
    score_map = y

    label_map = np.argmax(score_map, axis=0)
    return label_map, score_map


def test_single_case_patch(net, image, stride_xy, stride_z, patch_size, num_classes=1,save_un=False):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)
    pred_un = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()#.to(device) # cuda()
                if FLAGS.test_ml or FLAGS.test_vnet:
                    y_decoder = net(test_patch)
                    y_de_soft = F.softmax(y_decoder, dim=1)
                    y = y_de_soft
                    if FLAGS.save_un:
                        # enable the dropout to generate another prediction
                        net.apply(open_dropout)
                        y_decoder_1 = net(test_patch)
                        y_de_soft_1 = F.softmax(y_decoder_1, dim=1)
                        y_decoder_2 = net(test_patch)
                        y_de_soft_2 = F.softmax(y_decoder_2, dim=1)
                        y_decoder_3 = net(test_patch)
                        y_de_soft_3 = F.softmax(y_decoder_3, dim=1)
                        y_decoder_4 = net(test_patch)
                        y_de_soft_4 = F.softmax(y_decoder_4, dim=1)
                        y_decoder_5 = net(test_patch)
                        y_de_soft_5 = F.softmax(y_decoder_5, dim=1)# ([1, 2, x, y, z])
                        y_soft_fg = torch.cat((y_de_soft_1[:,1,...], y_de_soft_2[:,1,...],y_de_soft_3[:,1,...],
                        y_de_soft_4[:,1,...],y_de_soft_5[:,1,...]),dim=0)# ([5, x, y, z])
                else:
                    y_decoder, y_encoder = net(test_patch)
                    # print(y1.shape, out_dist.shape) # ([1, 2, x, y, z]) ([1, 1, x, y, z])
                    y_de_soft = F.softmax(y_decoder, dim=1)
                    y_en_soft = F.softmax(y_encoder, dim=1)
                
                    y = torch.mean(torch.stack([y_de_soft, y_en_soft]), dim=0)# ([1, 2, x, y, z]) 
                    if FLAGS.save_un:
                        net.apply(open_dropout)
                        y_decoder_1, y_encoder_1 = net(test_patch)
                        # print(y1.shape, out_dist.shape) # ([1, 2, x, y, z]) ([1, 1, x, y, z])
                        y_de_soft_1 = F.softmax(y_decoder_1, dim=1)
                        y_en_soft_1 = F.softmax(y_encoder_1, dim=1)
                        y_1 = torch.mean(torch.stack([y_de_soft_1, y_en_soft_1]), dim=0)# ([1, 2, x, y, z]) 
                        y_decoder_2, y_encoder_2 = net(test_patch)
                        y_de_soft_2 = F.softmax(y_decoder_2, dim=1)
                        y_en_soft_2 = F.softmax(y_encoder_2, dim=1)
                        y_2 = torch.mean(torch.stack([y_de_soft_2, y_en_soft_2]), dim=0)# ([1, 2, x, y, z]) 
                        y_decoder_3, y_encoder_3 = net(test_patch)
                        y_de_soft_3 = F.softmax(y_decoder_3, dim=1)
                        y_en_soft_3 = F.softmax(y_encoder_3, dim=1)
                        y_3 = torch.mean(torch.stack([y_de_soft_3, y_en_soft_3]), dim=0)# ([1, 2, x, y, z]) 
                        y_decoder_4, y_encoder_4 = net(test_patch)
                        y_de_soft_4 = F.softmax(y_decoder_4, dim=1)
                        y_en_soft_4 = F.softmax(y_encoder_4, dim=1)
                        y_4 = torch.mean(torch.stack([y_de_soft_4, y_en_soft_4]), dim=0)# ([1, 2, x, y, z]) 
                        y_decoder_5, y_encoder_5 = net(test_patch)
                        y_de_soft_5 = F.softmax(y_decoder_5, dim=1)
                        y_en_soft_5 = F.softmax(y_encoder_5, dim=1)
                        y_5 = torch.mean(torch.stack([y_de_soft_5, y_en_soft_5]), dim=0)# ([1, 2, x, y, z]) 
                        y_soft_fg = torch.cat((y_1[:,1,...],y_2[:,1,...],y_3[:,1,...],y_4[:,1,...],y_5[:,1,...]),dim=0)# ([5, x, y, z]) 
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1
                if save_un:
                    y_out_np = y_soft_fg.cpu().data.numpy()
                    mean_fg = np.mean(y_out_np, axis=0)
                    entropy = -mean_fg*np.log(mean_fg+1e-7)
                    pred_un[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] = entropy

    score_map = score_map/np.expand_dims(cnt,axis=0)
    pred_un = pred_un/cnt
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    if save_un:
        return label_map, score_map, pred_un
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt, space=0.625):
    dice = dc(pred, gt)
    jc_score = jc(pred, gt)

    asd_score = assd(pred, gt, voxelspacing=space)
    hd95_score = hd95(pred, gt, voxelspacing=space)
    hd_score = hd(pred, gt, voxelspacing=space)
    nsd = normalized_surface_dice(pred, gt, voxelspacing=space)
    return dice, jc_score, asd_score, hd95_score, hd_score, nsd

def normalized_surface_dice(pred, gt, voxelspacing=None):
    surface_dis = surf_metric.compute_surface_distances(gt.astype(np.bool),
                                                        pred.astype(np.bool),
                                                        spacing_mm=(voxelspacing, voxelspacing, voxelspacing))
    surface_dice = surf_metric.compute_surface_dice_at_tolerance(surface_dis, 1)
    return surface_dice


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert(labels.max() != 0) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def open_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def preprocess_test_data(image,label,output_size):

    # image, img_header = nrrd.read(image_path)
    # label, gt_header = nrrd.read(image_path.replace('lgemri.nrrd', 'laendo.nrrd'))
    # label = (label == 255).astype(np.uint8)
    w, h, d = label.shape

    tempL = np.nonzero(label)
    minx, maxx = np.min(tempL[0]), np.max(tempL[0])
    miny, maxy = np.min(tempL[1]), np.max(tempL[1])
    minz, maxz = np.min(tempL[2]), np.max(tempL[2])
    if maxx-minx >= output_size[0]:
        assert "Object out of range in x!!!!!!!!!!!"
        print("Object out of range in x!!!!!!!!!!!")
    if maxy-miny >= output_size[1]:
        assert "Object out of range in y!!!!!!!!!!!"
        print("Object out of range in y!!!!!!!!!!!")
    mid_x = (maxx-minx) // 2 + minx
    mid_y = (maxy-miny) // 2 + miny
    if mid_x - output_size[0]//2 <0:
        minx = 0
        maxx = output_size[0]
    elif mid_x + output_size[0]//2>w:
        maxx = w
        minx = w -output_size[0]
    else:
        minx = mid_x - output_size[0]//2
        maxx = mid_x + output_size[0]//2
    if mid_y - output_size[1]//2 <0:
        miny = 0
        maxy = output_size[1]
    elif mid_y + output_size[1]//2>h:
        maxy = h
        miny = h -output_size[1]
    else:
        miny = mid_y - output_size[1]//2
        maxy = mid_y + output_size[1]//2

    image = (image - np.mean(image)) / np.std(image)
    image = image.astype(np.float32)
    image = image[minx:maxx, miny:maxy]
    label = label[minx:maxx, miny:maxy]
    return image, label, minx, maxx, miny, maxy
if __name__ == '__main__':
    metric = test_calculate_metric(FLAGS.epoch_num)
    # print(metric)
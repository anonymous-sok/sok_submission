from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import cv2
learning_rate = 0.01 #0.07
epochs = 220


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))
# hit = 2
def allocation_strategy(frameRate):
    K = 1000
    next_reactivate = {}  # 下一次需要重新激活的时间
    max_active = 0  # 最大可用跟踪器数量
    strategy = []
    tracker_id = 0
    t = 0
    while(t<K):
        if t == 0:
            strategy.append(tracker_id)
            next_reactivate[tracker_id] = t+frameRate+1
            tracker_id += 1
            t+=1
        else:
            reactivate_time = min(next_reactivate.values())
            reactivate_tracker, reactivate_time = min(next_reactivate.items(), key=lambda x: x[1])
            if reactivate_time - t <= 1:
                strategy.append(reactivate_tracker)
                next_reactivate[reactivate_tracker] = t+frameRate+1
                t+=1
            else:
                strategy.append(tracker_id)
                strategy.append(tracker_id)
                next_reactivate[tracker_id] = t+frameRate+2
                tracker_id += 1
                t+=2
    return strategy, tracker_id

def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def run_attack(outputs,outputs_2,bx, strategy, max_tracker_num, adam_opt):

    per_num_b = (25*45)/max_tracker_num
    per_num_m = (50*90)/max_tracker_num
    per_num_s = (100*180)/max_tracker_num

    scores = outputs[:,5] * outputs[:,4]
    scores_2 = outputs_2[:,5] * outputs_2[:,4]
    sel_scores_b = scores[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_scores_m = scores[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_scores_s = scores[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]
    sel_scores_b_2 = scores_2[int(100*180+50*90+(strategy)*per_num_b):int(100*180+50*90+(strategy+1)*per_num_b)]
    sel_scores_m_2 = scores_2[int(100*180+(strategy)*per_num_m):int(100*180+(strategy+1)*per_num_m)]
    sel_scores_s_2 = scores_2[int((strategy)*per_num_s):int((strategy+1)*per_num_s)]

    sel_dets = torch.cat((sel_scores_b, sel_scores_m, sel_scores_s), dim=0)
    sel_dets_2 = torch.cat((sel_scores_b_2, sel_scores_m_2, sel_scores_s_2), dim=0)
    targets = torch.ones_like(sel_dets)
    loss1 = 10*(F.mse_loss(sel_dets, targets, reduction='sum')+F.mse_loss(sel_dets_2, targets, reduction='sum'))
    loss2 = 40*torch.norm(bx, p=2)
    targets = torch.ones_like(scores)
    loss3 = 1.0*(F.mse_loss(scores, targets, reduction='sum')+F.mse_loss(scores_2, targets, reduction='sum'))
    loss = loss1+loss3#+loss2
    
    loss.requires_grad_(True)
    adam_opt.zero_grad()
    loss.backward(retain_graph=True)
    
    # adam_opt.step()
    bx.grad = bx.grad / (torch.norm(bx.grad,p=2) + 1e-20)
    bx.data = -1.5 * bx.grad+ bx.data
    count = (scores > 0.3).sum()
    # print('loss',loss.item(),'loss_1',loss1.item(),'loss_2',loss2.item(),'loss_3',loss3.item(),'count:',count.item())
    return bx



class MOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.args = args

    def evaluate(
        self,
        model,
        device,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        frame_id = 0
        total_l1 = 0
        total_l2 = 0
        strategy = 0
        max_tracker_num = int(6)
        strategy, max_tracker_num = allocation_strategy(max_tracker_num)
        rgb_means=torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1).to(device)
        std=torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1).to(device)
        for cur_iter, (imgs,path,imgs_2,path_2) in enumerate(
            progress_bar(self.dataloader)
            ):
            print('strategy:',strategy[cur_iter])
            print(path,path_2)
            frame_id += 1
            bx = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]))
            bx = bx.astype(np.float32)
            bx = torch.from_numpy(bx).to(device).unsqueeze(0)
            bx = bx.data.requires_grad_(True)
            adam_opt = Adam([bx], lr=learning_rate, amsgrad=True)
            imgs = imgs.type(tensor_type)
            imgs = imgs.to(device)
            imgs_2 = imgs_2.type(tensor_type)
            imgs_2 = imgs_2.to(device)
            #(1,23625,6)
            
            for iter in tqdm(range(epochs)):
                added_imgs = imgs+bx
                added_imgs_2 = imgs_2+bx
                
                l2_norm = torch.sqrt(torch.mean(bx ** 2))
                l1_norm = torch.norm(bx, p=1)/(bx.shape[3]*bx.shape[2])
                added_imgs.clamp_(min=0, max=1)
                added_imgs_2.clamp_(min=0, max=1)
                input_imgs = (added_imgs - rgb_means)/std
                input_imgs_2 = (added_imgs_2 - rgb_means)/std
                if half:
                    input_imgs = input_imgs.half()
                    input_imgs_2 = input_imgs_2.half()
                outputs = model(input_imgs)[0]
                outputs_2 = model(input_imgs_2)[0]
                bx = run_attack(outputs,outputs_2,bx, strategy[cur_iter], max_tracker_num, adam_opt)

            # if strategy == max_tracker_num-1:
            #     strategy = 0
            # else:
            #     strategy += 1
            print(added_imgs.shape)
            added_blob = torch.clamp(added_imgs*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            added_blob = added_blob[..., ::-1]
            added_blob_2 = torch.clamp(added_imgs_2*255,0,255).squeeze().permute(1, 2, 0).detach().cpu().numpy()
            added_blob_2 = added_blob_2[..., ::-1]
            
            save_dir = path[0].replace("dataset", f"botsort_stra_{self.confthre}")
            save_dir_2 = path_2[0].replace("dataset", f"botsort_stra_{self.confthre}")
            result_dir = os.path.dirname(save_dir)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                print(f"data saved to {save_dir}")
            print(f"data saved to {save_dir}")
            cv2.imwrite(save_dir, added_blob)
            cv2.imwrite(save_dir_2, added_blob_2)
            print(l1_norm.item(),l2_norm.item())
            total_l1 += l1_norm
            total_l2 += l2_norm
            mean_l1 = total_l1/frame_id
            mean_l2 = total_l2/frame_id
            print(mean_l1.item(),mean_l2.item())
            del bx
            del adam_opt
            del outputs
            del outputs_2
            del imgs
            del imgs_2

        return mean_l1,mean_l2

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xywh2xyxy(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list


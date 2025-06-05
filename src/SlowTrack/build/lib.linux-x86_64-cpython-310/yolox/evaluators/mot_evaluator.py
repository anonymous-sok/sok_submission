from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch
import cv2

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.byte_tracker import BYTETracker
from yolox.sort_tracker.sort import Sort
from yolox.deepsort_tracker.deepsort import DeepSort
from yolox.motdt_tracker.motdt_tracker import OnlineTracker

import contextlib
import io
import os
import itertools
import json
import tempfile
import time

import motmetrics as mm
import numpy as np
import pandas as pd
from pathlib import Path

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


def write_results_with_scores(filename, results):
    """Write tracking results with scores in MOT format"""
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id, id=track_id, 
                    x1=round(x1, 1), y1=round(y1, 1), 
                    w=round(w, 1), h=round(h, 1), 
                    s=round(score, 2)
                )
                f.write(line)
    logger.info(f'Tracking results saved to {filename}')
    



import time


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.


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
        results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        n_samples = len(self.dataloader) - 1
        timer = Timer()
        stage_timer = Timer()
        stage_track_timer = Timer()
        max_stage_time = 0
        total_tracker = 0
        total_total_tracker = 0
        # max_track_num = 0
        total_track_time = 0
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        frame_id = 0
        for cur_iter, (imgs, path) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                self.args.track_buffer = 30
                self.args.track_thresh = ori_thresh

                frame_id += 1 
                if frame_id == 1:
                    tracker = BYTETracker(self.args)

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                timer.tic()
                stage_timer.tic()

                outputs = model(imgs)
                
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                count = (outputs[:,:,5]* outputs[:,:,4] > 0.3).sum()
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)[0]
                
                print(count)
                if outputs is not None:
                    print(len(outputs))

            tmp_img = cv2.imread(path[0])
            # run tracking
            if outputs is not None:
                online_targets,tmp_tracker, average_time = tracker.update(outputs, tmp_img.shape, self.img_size, stage_track_timer)
            else:
                tmp_tracker = 0
            timer.toc()
            stage_timer.toc()
            print('tracker_num:', tmp_tracker)
            total_tracker += tmp_tracker
            if (frame_id) % 20 == 0:
                if average_time>max_stage_time:
                    max_stage_time = average_time
                    max_track_num = total_tracker/20
                
                total_total_tracker += total_tracker
                print(max_stage_time,max_track_num)
                print(total_tracker/20)
                total_tracker = 0
                print('total Processing frame {} ({:.2f} fps)({}s)(total{}s)'.format(frame_id, 1. / max(1e-5, timer.average_time),timer.average_time,timer.total_time))
                print('stage Processing frame {} ({:.2f} fps)({}s)'.format(frame_id, 1. / max(1e-5, average_time),average_time))
                print(stage_track_timer.average_time)
                total_track_time += stage_track_timer.total_time
                stage_timer.clear()
                stage_track_timer.clear()
        print(total_total_tracker)
        print(total_track_time)
        print(timer.total_time)

        return total_total_tracker, total_track_time, timer.total_time

    def evaluate_sort(
        self,
        model,
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

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = Sort(self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = Sort(self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_deepsort(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
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

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
        
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = DeepSort(model_folder, min_confidence=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results_no_score(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            # save results
            results.append((frame_id, online_tlwhs, online_ids))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results_no_score(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def evaluate_motdt(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        model_folder=None
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

        inference_time = 0
        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                if frame_id == 1:
                    tracker = OnlineTracker(model_folder, min_cls_score=self.args.track_thresh)
                    if len(results) != 0:
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        write_results(result_filename, results)
                        results = []

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

            output_results = self.convert_to_coco_format(outputs, info_imgs, ids)
            data_list.extend(output_results)

            # run tracking
            online_targets = tracker.update(outputs[0], info_imgs, self.img_size, img_file_name[0])
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
            # save results
            results.append((frame_id, online_tlwhs, online_ids, online_scores))

            if is_time_record:
                track_end = time_synchronized()
                track_time += track_end - infer_end
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                write_results(result_filename, results)

        statistics = torch.cuda.FloatTensor([inference_time, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

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
            bboxes = xyxy2xywh(bboxes)

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

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
        

    def evaluate_mota(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        gt_folder=None
    ):
        """
        MOT Evaluation with MOTA calculation. Iterate inference on the test dataset
        and calculate MOT metrics including MOTA.

        Args:
            model : model to evaluate.
            gt_folder : path to ground truth annotations for MOTA calculation

        Returns:
            mota (float) : Multiple Object Tracking Accuracy
            motp (float) : Multiple Object Tracking Precision  
            summary (dict): detailed MOT metrics summary
        """
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        
        # Tracking results storage
        tracking_results = []
        video_names = defaultdict()
        progress_bar = tqdm if is_main_process() else iter

        # Performance tracking (keep your existing metrics)
        n_samples = len(self.dataloader) - 1
        timer = Timer()
        stage_timer = Timer()
        stage_track_timer = Timer()
        max_stage_time = 0
        total_tracker = 0
        total_total_tracker = 0
        total_track_time = 0
        
        # Setup result folder
        if result_folder is None:
            result_folder = "tracking_results"
        os.makedirs(result_folder, exist_ok=True)
        
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = BYTETracker(self.args)
        ori_thresh = self.args.track_thresh
        frame_id = 0
        
        for cur_iter, (imgs, path) in enumerate(progress_bar(self.dataloader)):
            with torch.no_grad():
                # init tracker
                self.args.track_buffer = 30
                self.args.track_thresh = ori_thresh

                frame_id += 1 
                if frame_id == 1:
                    tracker = BYTETracker(self.args)

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                timer.tic()
                stage_timer.tic()

                outputs = model(imgs)
                
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())
                count = (outputs[:,:,5]* outputs[:,:,4] > 0.3).sum()
                outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)[0]
                
                print(count)
                if outputs is not None:
                    print(len(outputs))

            tmp_img = cv2.imread(path[0])
            
            # run tracking
            if outputs is not None:
                online_targets, tmp_tracker, average_time = tracker.update(outputs, tmp_img.shape, self.img_size, stage_track_timer)
                
                # Extract tracking results for MOTA calculation
                online_tlwhs = []
                online_ids = []
                online_scores = []
                
                for target in online_targets:
                    if hasattr(target, 'tlwh'):  # ByteTracker format
                        tlwh = target.tlwh
                        track_id = target.track_id
                        score = target.score if hasattr(target, 'score') else 1.0
                    else:  # Alternative format
                        tlwh = [target[0], target[1], target[2] - target[0], target[3] - target[1]]
                        track_id = int(target[4])
                        score = target[5] if len(target) > 5 else 1.0
                    
                    # Filter small and vertical boxes
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(track_id)
                        online_scores.append(score)
                
                # Store results for this frame
                tracking_results.append((frame_id, online_tlwhs, online_ids, online_scores))
            else:
                tmp_tracker = 0
                # Store empty result for this frame
                tracking_results.append((frame_id, [], [], []))
                
            timer.toc()
            stage_timer.toc()
            print('tracker_num:', tmp_tracker)
            total_tracker += tmp_tracker
            
            if (frame_id) % 20 == 0:
                if average_time > max_stage_time:
                    max_stage_time = average_time
                    max_track_num = total_tracker/20
                
                total_total_tracker += total_tracker
                print(max_stage_time, max_track_num)
                print(total_tracker/20)
                total_tracker = 0
                print('total Processing frame {} ({:.2f} fps)({}s)(total{}s)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time), timer.average_time, timer.total_time))
                print('stage Processing frame {} ({:.2f} fps)({}s)'.format(
                    frame_id, 1. / max(1e-5, average_time), average_time))
                print(stage_track_timer.average_time)
                total_track_time += stage_track_timer.total_time
                stage_timer.clear()
                stage_track_timer.clear()
        
        print(total_total_tracker)
        print(total_track_time)
        print(timer.total_time)
        
        # Save tracking results to file
        result_filename = os.path.join(result_folder, 'tracking_results.txt')
        write_results_with_scores(result_filename, tracking_results)
        
        # Calculate MOTA if ground truth is provided
        if gt_folder is not None and os.path.exists(gt_folder):
            print("\n=== Calculating MOTA ===")
            mota, motp, summary = self.calculate_mota_from_results(tracking_results, gt_folder)
            
            print(f"\n=== MOT Evaluation Results ===")
            print(f"MOTA: {mota:.2f}%")
            print(f"MOTP: {motp:.2f}%")
            if summary is not None:
                print("\nDetailed Metrics:")
                print(summary)
            
            return mota, motp, summary
        else:
            print(f"Tracking results saved to: {result_filename}")
            print("Provide --gt_folder argument to calculate MOTA")
            return total_total_tracker, total_track_time, timer.total_time
        

    def calculate_mota_from_results(self, tracking_results, gt_folder):
        """
        Calculate MOTA from tracking results and ground truth
        
        Args:
            tracking_results: List of (frame_id, tlwhs, track_ids, scores)
            gt_folder: Path to ground truth folder
        
        Returns:
            mota, motp, summary
        """
        # Create MOT accumulator
        acc = mm.MOTAccumulator(auto_id=True)
        
        # Find ground truth file (assuming single sequence)
        gt_files = list(Path(gt_folder).glob('*.txt'))
        if not gt_files:
            print(f"No ground truth files found in {gt_folder}")
            return 0, 0, None
        
        gt_file = gt_files[0]  # Use first GT file found
        print(f"Using ground truth file: {gt_file}")
        
        # Load ground truth
        try:
            gt_data = pd.read_csv(gt_file, header=None,
                                names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis'])
            # Filter valid ground truth (class == 1 for person, conf != 0)
            gt_data = gt_data[(gt_data['class'] == 1) & (gt_data['conf'] != 0)]
        except:
            # Fallback for simpler format
            gt_data = pd.read_csv(gt_file, header=None,
                                names=['frame', 'id', 'x', 'y', 'w', 'h'])
        
        # Convert tracking results to frame-based dictionary
        track_dict = {}
        for frame_id, tlwhs, track_ids, scores in tracking_results:
            track_dict[frame_id] = {
                'tlwhs': np.array(tlwhs) if tlwhs else np.empty((0, 4)),
                'track_ids': np.array(track_ids) if track_ids else np.array([])
            }
        
        # Get all frames from both tracking and GT
        all_frames = set(track_dict.keys()) | set(gt_data['frame'].unique())
        
        # Process each frame
        for frame in sorted(all_frames):
            # Get tracking results for this frame
            if frame in track_dict:
                track_boxes = track_dict[frame]['tlwhs']
                track_ids = track_dict[frame]['track_ids']
            else:
                track_boxes = np.empty((0, 4))
                track_ids = np.array([])
            
            # Get ground truth for this frame  
            gt_frame = gt_data[gt_data['frame'] == frame]
            if len(gt_frame) > 0:
                gt_boxes = gt_frame[['x', 'y', 'w', 'h']].values
                gt_ids = gt_frame['id'].values
            else:
                gt_boxes = np.empty((0, 4))
                gt_ids = np.array([])
            
            # Convert tlwh to tlbr for motmetrics
            if len(track_boxes) > 0:
                track_boxes_br = track_boxes.copy()
                track_boxes_br[:, 2:] += track_boxes_br[:, :2]
            else:
                track_boxes_br = np.empty((0, 4))
                
            if len(gt_boxes) > 0:
                gt_boxes_br = gt_boxes.copy()
                gt_boxes_br[:, 2:] += gt_boxes_br[:, :2]
            else:
                gt_boxes_br = np.empty((0, 4))
            
            # Calculate distances (IoU-based)
            if len(gt_boxes_br) > 0 and len(track_boxes_br) > 0:
                distances = mm.distances.iou_matrix(gt_boxes_br, track_boxes_br, max_iou=0.5)
            else:
                distances = np.empty((len(gt_boxes_br), len(track_boxes_br)))
            
            # Update accumulator
            acc.update(gt_ids, track_ids, distances)
        
        # Calculate metrics
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'num_switches', 
                                        'idf1', 'idp', 'idr', 'recall', 'precision'], 
                            name='overall')
        
        # Extract key metrics
        mota = summary['mota'].iloc[0] * 100 if 'mota' in summary.columns and len(summary) > 0 else 0
        motp = summary['motp'].iloc[0] * 100 if 'motp' in summary.columns and len(summary) > 0 else 0
        
        return mota, motp, summary
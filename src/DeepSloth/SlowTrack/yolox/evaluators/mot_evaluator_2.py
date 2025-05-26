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

import numpy as np
from datetime import datetime

class AccuracyMetrics:
    """
    Accuracy metrics calculator for MOT evaluation
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_detections = 0
        self.total_ground_truth = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.id_switches = 0
        self.track_fragments = 0
        self.mostly_tracked = 0
        self.partially_tracked = 0
        self.mostly_lost = 0
        
        # For tracking accuracy
        self.gt_trajectories = {}  # ground truth trajectories
        self.pred_trajectories = {}  # predicted trajectories
        self.matched_pairs = {}  # matched GT-Pred pairs
        
        # For mAP calculation
        self.all_predictions = []  # List to store all predictions with scores
        self.all_ground_truths = []  # List to store all ground truths
        self.frame_predictions = {}  # Frame-wise predictions for mAP
        self.frame_ground_truths = {}  # Frame-wise ground truths for mAP
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Convert to (x1, y1, x2, y2) format
        box1_coords = [x1, y1, x1 + w1, y1 + h1]
        box2_coords = [x2, y2, x2 + w2, y2 + h2]
        
        # Calculate intersection
        xi1 = max(box1_coords[0], box2_coords[0])
        yi1 = max(box1_coords[1], box2_coords[1])
        xi2 = min(box1_coords[2], box2_coords[2])
        yi2 = min(box1_coords[3], box2_coords[3])
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
            
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_ap(self, predictions, ground_truths, iou_threshold=0.5):
        """
        Calculate Average Precision (AP) for a single IoU threshold
        
        Args:
            predictions: List of (bbox, score) tuples
            ground_truths: List of bbox tuples
            iou_threshold: IoU threshold for positive detection
            
        Returns:
            Average Precision value
        """
        if len(predictions) == 0:
            return 0.0
        if len(ground_truths) == 0:
            return 0.0
            
        # Sort predictions by confidence score (descending)
        predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)
        
        # Track which ground truths have been matched
        gt_matched = [False] * len(ground_truths)
        
        true_positives = []
        false_positives = []
        
        for pred_bbox, pred_score in predictions_sorted:
            # Find best matching ground truth
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_bbox in enumerate(ground_truths):
                if gt_matched[gt_idx]:
                    continue
                    
                iou = self.calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if it's a true positive or false positive
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                true_positives.append(1)
                false_positives.append(0)
                gt_matched[best_gt_idx] = True
            else:
                true_positives.append(0)
                false_positives.append(1)
        
        # Calculate precision and recall arrays
        tp_cumsum = np.cumsum(true_positives)
        fp_cumsum = np.cumsum(false_positives)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Add endpoints for interpolation
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))
        
        # Compute the precision envelope (monotonically decreasing)
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
            
        return ap
    
    def calculate_map(self, iou_thresholds=None):
        """
        Calculate mean Average Precision (mAP) across different IoU thresholds
        
        Args:
            iou_thresholds: List of IoU thresholds to evaluate
            
        Returns:
            Dictionary containing mAP metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75]  # Common thresholds, can extend to [0.5:0.05:0.95]
        
        # Collect all predictions and ground truths across all frames
        all_predictions = []
        all_ground_truths = []
        
        for frame_id in self.frame_predictions:
            frame_preds = self.frame_predictions.get(frame_id, [])
            frame_gts = self.frame_ground_truths.get(frame_id, [])
            
            # Add frame predictions (bbox, score)
            for pred in frame_preds:
                bbox, score = pred[0], pred[2] if len(pred) > 2 else 1.0
                all_predictions.append((bbox, score))
            
            # Add frame ground truths (bbox only)
            for gt in frame_gts:
                bbox = gt[0]
                all_ground_truths.append(bbox)
        
        # Calculate AP for each IoU threshold
        aps = []
        ap_results = {}
        
        for iou_thresh in iou_thresholds:
            ap = self.calculate_ap(all_predictions, all_ground_truths, iou_thresh)
            aps.append(ap)
            ap_results[f'AP@{iou_thresh:.2f}'] = ap
        
        # Calculate mAP
        map_score = np.mean(aps) if aps else 0.0
        
        ap_results['mAP'] = map_score
        ap_results['AP@0.5'] = aps[0] if len(aps) > 0 else 0.0
        if len(aps) > 1:
            ap_results['AP@0.75'] = aps[1]
        
        return ap_results
    
    def update(self, pred_results, gt_results, frame_id, iou_threshold=0.5):
        """
        Update metrics with prediction and ground truth results for a frame
        
        Args:
            pred_results: List of (tlwh, track_id, score) tuples for predictions
            gt_results: List of (tlwh, track_id) tuples for ground truth
            frame_id: Current frame ID
            iou_threshold: IoU threshold for matching
        """
        if pred_results is None:
            pred_results = []
        if gt_results is None:
            gt_results = []
            
        # Store frame-wise data for mAP calculation
        self.frame_predictions[frame_id] = pred_results
        self.frame_ground_truths[frame_id] = gt_results
        
        # Extract bounding boxes and IDs
        pred_boxes = [item[0] for item in pred_results] if pred_results else []
        pred_ids = [item[1] for item in pred_results] if pred_results else []
        pred_scores = [item[2] if len(item) > 2 else 1.0 for item in pred_results] if pred_results else []
        
        gt_boxes = [item[0] for item in gt_results] if gt_results else []
        gt_ids = [item[1] for item in gt_results] if gt_results else []
        
        # Update trajectory tracking
        for i, gt_id in enumerate(gt_ids):
            if gt_id not in self.gt_trajectories:
                self.gt_trajectories[gt_id] = []
            self.gt_trajectories[gt_id].append((frame_id, gt_boxes[i]))
            
        for i, pred_id in enumerate(pred_ids):
            if pred_id not in self.pred_trajectories:
                self.pred_trajectories[pred_id] = []
            self.pred_trajectories[pred_id].append((frame_id, pred_boxes[i]))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, pred_box in enumerate(pred_boxes):
                iou_matrix[i, j] = self.calculate_iou(gt_box, pred_box)
        
        # Hungarian matching (simplified greedy approach)
        matched_gt = set()
        matched_pred = set()
        matches = []
        
        # Sort by IoU in descending order
        iou_pairs = []
        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                if iou_matrix[i, j] >= iou_threshold:
                    iou_pairs.append((iou_matrix[i, j], i, j))
        
        iou_pairs.sort(reverse=True)
        
        for iou_val, i, j in iou_pairs:
            if i not in matched_gt and j not in matched_pred:
                matches.append((i, j))
                matched_gt.add(i)
                matched_pred.add(j)
        
        # Update metrics
        self.true_positives += len(matches)
        self.false_positives += len(pred_boxes) - len(matches)
        self.false_negatives += len(gt_boxes) - len(matches)
        
        self.total_detections += len(pred_boxes)
        self.total_ground_truth += len(gt_boxes)
        
        # Track ID consistency
        for gt_idx, pred_idx in matches:
            gt_id = gt_ids[gt_idx]
            pred_id = pred_ids[pred_idx]
            
            if gt_id in self.matched_pairs:
                if self.matched_pairs[gt_id] != pred_id:
                    self.id_switches += 1
                    self.matched_pairs[gt_id] = pred_id
            else:
                self.matched_pairs[gt_id] = pred_id
    
    def calculate_accuracy_metrics(self, include_map=True):
        """Calculate various accuracy metrics including mAP"""
        metrics = {}
        
        # Basic detection metrics
        if self.true_positives + self.false_positives > 0:
            precision = self.true_positives / (self.true_positives + self.false_positives)
        else:
            precision = 0.0
            
        if self.true_positives + self.false_negatives > 0:
            recall = self.true_positives / (self.true_positives + self.false_negatives)
        else:
            recall = 0.0
            
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        # MOT-specific metrics
        if self.total_ground_truth > 0:
            mota = 1 - (self.false_negatives + self.false_positives + self.id_switches) / self.total_ground_truth
        else:
            mota = 0.0
            
        # Tracking accuracy
        if self.true_positives > 0:
            motp = self.true_positives / (self.true_positives + self.false_positives + self.false_negatives)
        else:
            motp = 0.0
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mota': mota,  # Multiple Object Tracking Accuracy
            'motp': motp,  # Multiple Object Tracking Precision
            'id_switches': self.id_switches,
            'total_detections': self.total_detections,
            'total_ground_truth': self.total_ground_truth,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives
        })
        # import pdb; pdb.set_trace()
        # Add mAP metrics if requested
        if True:
            map_metrics = self.calculate_map()
            metrics.update(map_metrics)
        
        return metrics
    
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
        
        self.accuracy_metrics = AccuracyMetrics()


    def evaluate_with_accuracy(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None,
        gt_data=None,
    ):
        """
        Evaluate tracking performance with accuracy metrics
        
        Args:
            model: model to evaluate
            gt_data: ground truth data in format {frame_id: [(tlwh, track_id), ...]}
            Other args: same as original evaluate method
            
        Returns:
            Dictionary containing tracking metrics and accuracy scores
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
        total_track_time = 0
        
        # Reset accuracy metrics
        self.accuracy_metrics.reset()
        
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
        
        clean_run_data = {}
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
                
                # print(f"count: {count}")
                # if outputs is not None:
                #     print(f"len of count: {len(outputs)}")
            
            tmp_img = cv2.imread(path[0])
            
            # run tracking
            pred_results = []
            if outputs is not None:
                online_targets, tmp_tracker, average_time = tracker.update(outputs, tmp_img.shape, self.img_size, stage_track_timer)
                
                # Extract prediction results for accuracy calculation
                for target in online_targets:
                    if hasattr(target, 'tlwh') and hasattr(target, 'track_id') and hasattr(target, 'score'):
                        pred_results.append((target.tlwh, target.track_id, target.score))
                    elif hasattr(target, 'tlbr') and hasattr(target, 'track_id'):
                        # Convert tlbr to tlwh if needed
                        tlbr = target.tlbr
                        tlwh = [tlbr[0], tlbr[1], tlbr[2] - tlbr[0], tlbr[3] - tlbr[1]]
                        score = getattr(target, 'score', 1.0)
                        pred_results.append((tlwh, target.track_id, score))
            else:
                tmp_tracker = 0
            
            # Get ground truth for current frame
            # Update accuracy metrics
            
            clean_run_data[frame_id] = pred_results
            
            # import pdb; pdb.set_trace()
            if gt_data is not None:
                gt_results = gt_data.get(frame_id, [])
                self.accuracy_metrics.update(pred_results, gt_results, frame_id)
            
            timer.toc()
            stage_timer.toc()
            # print('tracker_num:', tmp_tracker)
            total_tracker += tmp_tracker
            
            if (frame_id) % 20 == 0:
                if average_time > max_stage_time:
                    max_stage_time = average_time
                    max_track_num = total_tracker/20
                
                total_total_tracker += total_tracker
                # print(max_stage_time, max_track_num)
                # print(total_tracker/20)
                total_tracker = 0
                # print('total Processing frame {} ({:.2f} fps)({}s)(total{}s)'.format(
                #     frame_id, 1. / max(1e-5, timer.average_time), timer.average_time, timer.total_time))
                # print('stage Processing frame {} ({:.2f} fps)({}s)'.format(
                #     frame_id, 1. / max(1e-5, average_time), average_time))
                # print(stage_track_timer.average_time)
                total_track_time += stage_track_timer.total_time
                stage_timer.clear()
                stage_track_timer.clear()
                
        
        # Calculate final accuracy metrics
        accuracy_results = self.accuracy_metrics.calculate_accuracy_metrics()
        
        print(f"total_total_tracker: {total_total_tracker}")
        print(f"total_track_time: {total_track_time}")
        print(f"total_time: {timer.total_time}")
        
        # Print accuracy metrics
        print("\n=== Accuracy Metrics ===")
        print(f"Precision: {accuracy_results['precision']:.4f}")
        print(f"Recall: {accuracy_results['recall']:.4f}")
        print(f"F1-Score: {accuracy_results['f1_score']:.4f}")
        print(f"MOTA: {accuracy_results['mota']:.4f}")
        print(f"MOTP: {accuracy_results['motp']:.4f}")
        print(f"mAP: {accuracy_results['mAP']:.4f}")
        print(f"mAP50: {accuracy_results['AP@0.5']:.4f}")
        print(f"ID Switches: {accuracy_results['id_switches']}")
        print(f"True Positives: {accuracy_results['true_positives']}")
        print(f"False Positives: {accuracy_results['false_positives']}")
        print(f"False Negatives: {accuracy_results['false_negatives']}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"/opt/dlami/nvme/SlowTrack/results/{timestamp}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        ret =  {
            'tracking_metrics': {
                'total_tracker': total_total_tracker,
                'total_track_time': total_track_time,
                'total_time': timer.total_time
            },
            'accuracy_metrics': accuracy_results
        }
        
        print(f"Saving results to {output_path}")
        results = {**ret, **accuracy_results}
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        return ret


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
                
                print(f"count: {count}")
                if outputs is not None:
                    print(f"len of count: {len(outputs)}")
            
            # import pdb; pdb.set_trace()
            
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
        print(f"total_total_tracker: {total_total_tracker}")
        print(f"total_track_time: {total_track_time}")
        print(f"total_time: {timer.total_time}")

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
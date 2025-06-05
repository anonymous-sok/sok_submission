

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
        
        # Add mAP metrics if requested
        if True:
            map_metrics = self.calculate_map()
            metrics.update(map_metrics)
        
        return metrics
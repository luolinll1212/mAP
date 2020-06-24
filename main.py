# *_*coding:utf-8 *_*
import os
import numpy as np

def compute_ap(recall, precision):

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def bbox_iou_numpy(box1, box2):
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0]
    )
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1]
    )

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) *
                        (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def evaluate(num_classes, all_detections, all_annotations, iou_thres=.5):
    """
    Code originally from https://github.com/eriklindernoren/PyTorch-YOLOv3.
    """

    average_precisions = {}
    for label in range(num_classes):
        true_positives = []
        scores = []
        num_annotations = 0

        for i in range(len(all_annotations)):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]

            num_annotations += annotations.shape[0]
            detected_annotations = []

            for *bbox, score in detections:
                scores.append(score)

                if annotations.shape[0] == 0:
                    true_positives.append(0)
                    continue

                overlaps = bbox_iou_numpy(
                    np.expand_dims(bbox, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_thres and assigned_annotation not in detected_annotations:
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        true_positives = np.array(true_positives)
        false_positives = np.ones_like(true_positives) - true_positives

        indices = np.argsort(-np.array(scores))
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        recall = true_positives / num_annotations
        precision = true_positives / \
            np.maximum(true_positives + false_positives,
                       np.finfo(np.float64).eps)

        average_precision = compute_ap(recall, precision)
        average_precisions[label] = average_precision

    return average_precisions

if __name__ == '__main__':
    classes = ['backpack', 'bed', 'book', 'bookcase', 'bottle', 'bowl', 'cabinetry', 'chair', 'coffeetable', 'countertop', 'cup', 'diningtable', 'doll', 'door', 'heater', 'nightstand', 'person', 'pictureframe', 'pillow', 'pottedplant', 'remote', 'shelf', 'sink', 'sofa', 'tap', 'tincan', 'tvmonitor', 'vase', 'wastecontainer', 'windowblind']
    num_classes = len(classes)

    all_detections = []
    all_groundtruths = []

    gt_path = "./input/ground-truth"
    detect_path = "./input/detection-results"

    for file in os.listdir(gt_path):
        # 检测数据
        with open(os.path.join(detect_path, file), "r", encoding="utf-8") as fread:
            lines = fread.readlines()

        table = [[] for _ in range(num_classes)]
        for line in lines:
            line = line.strip().split()
            label = line[0]
            index = classes.index(label)
            boxes = []
            boxes.append([line[2], line[3], line[4], line[5], line[1]]) # 二维
            table[index] = boxes

        # 预测框
        all_detections.append([np.array([]) for _ in range(num_classes)])
        for i in range(num_classes):
            all_detections[-1][i] = np.array(table[i], dtype=np.float32)

        # ground_truth
        with open(os.path.join(gt_path, file), "r", encoding="utf-8") as fread:
            lines = fread.readlines()

        table = [[] for _ in range(num_classes)]
        for line in lines:
            line = line.strip().split()
            label = line[0]
            index = classes.index(label)
            boxes = []
            boxes.append([line[1], line[2], line[3], line[4]]) # 二维
            table[index] = boxes

        # 预测框
        all_groundtruths.append([np.array([]) for _ in range(num_classes)])
        for i in range(num_classes):
            all_groundtruths[-1][i] = np.array(table[i], dtype=np.float32)

    # 计算mAP
    average_precisions = evaluate(
        num_classes, all_detections, all_groundtruths)

    for c, ap in average_precisions.items():
        print(f"Class '{c}' - AP: {ap}")

    mAP = np.mean(list(average_precisions.values()))
    print(f"mAP: {mAP}")
    print(f"""{"-"*40}end{"-"*40}""")
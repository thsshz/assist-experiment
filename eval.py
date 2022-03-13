import json
import argparse
import numpy as np


def compute_overlap(boxes, query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(detection_file):
    with open(detection_file, 'r') as f:
        lines = f.readlines()
        last_line = lines[len(lines) - 1]
        objects = last_line.replace('\n', '').split(' ')
        frame_num = int(objects[0]) + 1
        all_detections = [[None for i in range(2)] for j in range(frame_num)]
        for line in lines:
            objects = line.replace('\n', '').split(' ')
            frame_number = int(objects[0])
            xmin = float(objects[1])
            ymin = float(objects[2])
            xmax = float(objects[3])
            ymax = float(objects[4])
            label = int(objects[5]) - 1
            score = float(objects[6])
            bbox = [xmin, ymin, xmax, ymax, score]
            if all_detections[frame_number][label] is None:
                all_detections[frame_number][label] = np.array(bbox, ndmin=2)
            else:
                all_detections[frame_number][label] = np.append(all_detections[frame_number][label], np.array(bbox, ndmin=2), axis=0)
    return all_detections


def _get_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        last_line = lines[len(lines) - 1]
        objects = last_line.replace('\n', '').split(' ')
        frame_num = int(objects[0]) + 1
        all_annotations = [[None for i in range(2)] for j in range(frame_num)]
        for line in lines:
            objects = line.replace('\n', '').split(' ')
            frame_number = int(objects[0])
            xmin = float(objects[1])
            ymin = float(objects[2])
            xmax = float(objects[3])
            ymax = float(objects[4])
            label = int(objects[5]) - 1
            bbox = [xmin, ymin, xmax, ymax]
            if all_annotations[frame_number][label] is None:
                all_annotations[frame_number][label] = np.array(bbox, ndmin=2)
            else:
                all_annotations[frame_number][label] = np.append(all_annotations[frame_number][label], np.array(bbox, ndmin=2), axis=0)
    return all_annotations


def evaluate(all_detections, all_annotations, iou_threshold):
    quata = {}

    for label in range(2):

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_annotations)):
            annotations = all_annotations[i][label]
            if annotations is not None:
                num_annotations += annotations.shape[0]
            if i >= len(all_detections):
                continue
            detections = all_detections[i][label]
            if detections is None or annotations is None:
                continue
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])
                if annotations is None:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute quata (F1 score and AP)
        F1_score = 2 * (precision[-1] * recall[-1]) / (precision[-1] + recall[-1])
        AP = _compute_ap(recall, precision)
        quata[label] = F1_score, AP, num_annotations
        print("label {:d} result:".format(label))
        print("annotations bbox num is {:d}".format(int(num_annotations)))
        print("detections bbox num is {:d}".format(len(precision)))
        print("precision is {:.5f}".format(precision[-1]))
        print("recall is {:.5f}".format(recall[-1]))
        print("F1 score is {:.5f}".format(F1_score))
        print("AP is {:.5f}".format(AP))
    mean_F1_score = 0.0
    mean_AP = 0.0
    total_num = 0
    for label in quata.keys():
        mean_F1_score += quata[label][0] * quata[label][2]
        mean_AP += quata[label][1] * quata[label][2]
        total_num += quata[label][2]
    mean_F1_score /= total_num
    mean_AP /= total_num
    print("mean F1 score is {:.5f}".format(mean_F1_score))
    print("mean AP is {:.5f}".format(mean_AP))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--detection", help="detection file name",
                        default="detection.txt")
    parser.add_argument("-a", "--annotation", help="annotation file name",
                        default="annotation.txt")
    args = parser.parse_args()
    iou_threshold = 0.5
    all_detections = _get_detections(args.detection)
    all_annotations = _get_annotations(args.annotation)
    evaluate(all_detections, all_annotations, iou_threshold)


if __name__ == "__main__":
    main()

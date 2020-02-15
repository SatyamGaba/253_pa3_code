import numpy as np

def iou(pred, labels):
    intersections, unions = [], []
    bad_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18, 29,30]
    n_class = 34
    for cls in range(n_class):
        if cls not in bad_ids:
            TP = ((pred==cls) & (labels==cls)).sum()
            FP = ((pred==cls) & (labels!=cls)).sum()
            FN = ((pred!=cls) & (labels==cls)).sum()
            # Complete this function
            intersection = TP
            union = (TP+FP+FN)
            if union == 0:
                intersections.append(0)
                unions.append(0)
                # if there is no ground truth, do not include in evaluation
            else:
                intersections.append(intersection)
                unions.append(union)
                # Append the calculated IoU to the list ious
    return intersections, unions

def pixel_acc(pred, target):
    pass
    #Complete this function
    # included in main

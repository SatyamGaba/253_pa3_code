import numpy as np
def iou(pred, target):
   
    print(pred.shape)
    pred = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    bad_ids = [0,1,2,3,4,5,6,9,10,14,15,16,18, 29,30]
    out = np.ones((len(pred), 34-len(bad_ids)))
    number_of_images = int(len(pred))
    n_class = 34
    for i in range(number_of_images):        
        j = 0        
        for cls in range(n_class):          
            if (cls not in bad_ids):       
                p = np.argmax(pred[i], axis = 0)
                t = np.argmax(target[i], axis = 0)
                               
                classy = np.ones(p.shape) * cls
                TP = np.where((p==classy) & (p==t),1,0).sum()
                FP = np.where((p==classy) & (p!=t),1,0).sum()
                FN = np.where((p!=classy) & (classy==t),1,0).sum()
                print(TP, FP, FN)
                intersection = TP # intersection calculation
                union = TP + FP + FN # Union calculation
                if union == 0:
                    out[i,j] = 0
                    j= j+1
                    # if there is no ground truth, do not include in evaluation
                else:
                    out[i,j] = intersection/union
                    j = j+1
                    #ious.append(intersection/union)# Append the calculated IoU to the list ious
        return out


def pixel_acc(pred, target):
    pass
    #Complete this function
    # included in main

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, \
                        balanced_accuracy_score, f1_score

def compute_score_metrics(true:np.ndarray, pred:np.ndarray, exclude_self=False, lag=False):
    assert np.shape(true)==np.shape(pred)
    if np.max(true) == np.min(true): #can't compute metrics
            return None,None
    if exclude_self:
        if lag: #lag inference
            dim,lag,_ = true.shape
            self_ind = np.eye(dim)[:,np.newaxis,:] * np.ones((dim,lag,dim))
        else: #variable inference
            self_ind = np.eye(true.shape[0])
        true = true[np.logical_not(self_ind)].flatten()
        pred = pred[np.logical_not(self_ind)].flatten()

        auroc = roc_auc_score(true,pred)
        auprc = average_precision_score(true,pred)
    else:
        auroc = roc_auc_score(true.flatten(),pred.flatten())
        auprc = average_precision_score(true.flatten(),pred.flatten())
    return auroc, auprc

def compute_binary_metrics(true:np.ndarray, pred:np.ndarray, exclude_self=False, lag=False):
    assert np.shape(true)==np.shape(pred)
    if np.max(true)==np.min(true):
        return None,None,None
    if exclude_self:
        if lag: #lag inference
            dim,lag,_ = true.shape
            self_ind = np.eye(dim)[:,np.newaxis,:] * np.ones((dim,lag,dim))
        else: #variable inference
            self_ind = np.eye(true.shape[0])
        true = true[np.logical_not(self_ind)]
        pred = pred[np.logical_not(self_ind)]
    accuracy = accuracy_score(true.flatten(), pred.flatten())
    bal_accuracy = balanced_accuracy_score(true.flatten(), pred.flatten())
    fscore = f1_score(true.flatten(), pred.flatten())
    return accuracy, bal_accuracy, fscore

def compute_sign_sensitivity(true:np.ndarray, pred:np.ndarray, exclude_self=True, lag=False):
    assert np.shape(true)==np.shape(pred)
    if np.max(true)==np.min(true):
        return None,None,None
    if exclude_self:
        if lag: #lag inference
            dim,lag,_ = true.shape
            self_ind = np.eye(dim)[:,np.newaxis,:] * np.ones((dim,lag,dim))
        else: #variable inference
            self_ind = np.eye(true.shape[0])
        true = true[np.logical_not(self_ind)].flatten()
        pred = pred[np.logical_not(self_ind)].flatten()
    #sensitivity(+)
    all_pos = (true>0).astype(int)
    inferred_pos = (pred>0).astype(int)
    correct_pos = inferred_pos * all_pos 
    sensitivity_pos = correct_pos.sum()/all_pos.sum()
    #sensitivity(-)
    all_neg = (true<0).astype(int)
    inferred_neg = (pred<0).astype(int)
    correct_neg = inferred_neg * all_neg 
    sensitivity_neg = correct_neg.sum()/all_neg.sum()
    return sensitivity_pos, sensitivity_neg
import keras
import numpy as np
#check weather pathos is correctly installed
from multiprocessing import Pool
import pandas as pd
from augment import *

import numpy as np
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects),range=((0,1),(0,1)))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects,range=(0,1))[0]
    area_pred = np.histogram(y_pred, bins = pred_objects,range=(0,1))[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9
    intersection[intersection == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

def iou_metric_batch_nomean(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return metric


def list_split(mlist,num):
    assert type(mlist) == list,'mlist must be list!'
    assert type(num) == type(1),'num must be int!'
    assert num >= 1,'num must larger or equal to 1'
    l = len(mlist)
    num = min(num,l)
    left = l % num
    each_num = l // num
    res = []
    for i in range(num-1):
        res.append(mlist[i*each_num:(i+1)*each_num])
    res.append(mlist[(num-1)*each_num:])
    return res
def list_concat(lists):
    res = []
    for x in lists:
        res += x
    return res
def parallel_param_split(args,nthread,put_idx):
    def m_split(arg,nthread):
        if type(arg) == list:
            return list_split(arg,nthread)
        else:
            return np.array_split(arg,nthread)
    if put_idx:
        args = [[[i] for i in range(nthread)]]+[m_split(arg,nthread) for arg in args]
    else:
        args = [m_split(arg,nthread) for arg in args]
    
    return args
def parallel_run(func,args,nthread=1,put_idx=False):
    assert type(args) == list,'parallel error! params should be list'
    for arg in args:
        assert type(arg) in [list,np.ndarray,pd.DataFrame], 'parallel error! all params should be list, pd.DataFrame or numpy array! got %s'%(str(type(arg)))
        assert len(arg) == len(args[0]),'parallel error! all params should have same length'
        assert len(arg) != 0,'parallel error! params length should not be 0'
    nthread = min(nthread,len(args[0]))
    p = Pool(processes=nthread)
    args = parallel_param_split(args,nthread=1,put_idx=False)
    ret = p.map(func,list(zip(args[0][0],args[1][0])))
    p.close(); p.join();
    return ret

def eval_F1(model,X_val,y_val,cls_res=None):
    val_pred = tta_pred(model,X_val)
    val_pred = val_pred[:,13:-14,13:-14,:]
    val_true = y_val[:,13:-14,13:-14,:]
    thresholds = np.linspace(-0.15, 0.95, 55)
    def get_intval_pred_1(val_pred,thresholds):
        return np.int32(val_pred > thresholds)
    get_intval_pred = get_intval_pred_1
    #from tqdm import tqdm_notebook
    
    input_true = [val_true for threshold in thresholds]
    input_pred = [get_intval_pred(val_pred,threshold) for threshold in thresholds]
    #ious = np.array([iou_metric_batch(val_true, val_pred) for val_true,val_pred in (input_true,input_pred)])
    ious = parallel_run(multi_iou_metric_batch,args=[input_true,input_pred],nthread=6)
    
    #ious = np.array([iou_metric_batch(val_true, get_intval_pred(val_pred,threshold)) for threshold in thresholds])

    threshold_best_index = np.argmax(ious)
    iou_best = ious[threshold_best_index]
    threshold_best = thresholds[threshold_best_index]

    if not cls_res is None:
        def process(val_pred,cls_res):
            ret = np.copy(val_pred)
            ret[np.where(cls_res)] = 0
            return ret
        iou_best_cls = 0
        ths = np.linspace(0.9,1.0,20)
        input_true = [val_true for th in ths]
        input_pred = [get_intval_pred(process(val_pred,np.int32(cls_res>th)),threshold_best) for th in ths]
        cls_ious = parallel_run(multi_iou_metric_batch,args=[input_true,input_pred],nthread=6)
        cls_threshold_best_index = np.argmax(cls_ious)
        cls_iou_best = cls_ious[cls_threshold_best_index]
        cls_threshold_best = ths[cls_threshold_best_index]
        
        return 'threshold_best:%.2f iou_best:%f iou_th_best:%f iou_best_cls:%f'%(threshold_best,iou_best,cls_threshold_best,cls_iou_best),iou_best
    
    return 'threshold_best:%.2f iou_best:%f'%(threshold_best,iou_best),iou_best

class feval(keras.callbacks.Callback):
    def __init__(self,feval_func,X_val,y_val,val_steps = None,eval_best_only=True,monitor='val_loss',eval_th = 1e100, save_best_weight = False, save_model_name = None, feval_args = {}):
        super(feval, self).__init__()
        self.feval_func = feval_func
        self.X_val = X_val
        self.y_val = y_val
        self.eval_best_only = eval_best_only
        self.save_best_weight = save_best_weight
        self.monitor = monitor
        self.save_model_name = save_model_name
        self.best_loss = 1e100
        self.best_metric = 0
        self.eval_th = eval_th
        self.feval_args = feval_args
        self.last_save_name = ''
        if type(self.X_val) == list:
            assert val_steps is not None, 'param val_steps could not be None when X_val is generator'
        self.val_steps = val_steps
    def on_epoch_end(self, batch, logs={}):
        eval_this = True
        if self.eval_best_only:
            if logs[self.monitor] > self.best_loss:
                eval_this = False
            else:
                self.best_loss = logs[self.monitor]
        if logs[self.monitor] > self.eval_th:
            eval_this = False
        if eval_this:
            if type(self.X_val) == list:
                X_val = self.X_val[0](*self.X_val[1])
                val_pred = self.model.predict_generator(X_val,steps=self.val_steps)
            else:
                val_pred = self.model.predict(self.X_val,batch_size=32)
            resstr,res = self.feval_func[1](self.model,self.X_val,self.y_val,**self.feval_args)

            if res > self.best_metric:
                self.best_metric = res
                if self.save_best_weight:
                    print('save weight %f'%(res))
                    if self.last_save_name != '':
                        import os
                        os.remove(self.last_save_name)
                    self.model.save_weights(self.save_model_name.replace('.hdf5','%.6f.hdf5'%(res)),overwrite=True)
                    self.last_save_name = self.save_model_name.replace('.hdf5','%.6f.hdf5'%(res))
                    open('model_name.txt','w').write(self.last_save_name)
            print('------------------------------------------------------------------------------')
            print('|evaluate %s %s'%(self.feval_func[0],resstr))
            print('------------------------------------------------------------------------------')
def tta_pred(model,X):
    y = []
    for augment,args in [[aug_ori,{}],[aug_flip,{'t':1}]]:
        nx,augments = get_augmentation_data(X,augment,args)
        ny = model.predict(nx,batch_size=32)
        ny = get_deaugmentation_data(ny,augments)
        #print(ny.shape)
        y.append(ny)
    return np.mean(np.array(y),axis=0)
def multi_iou_metric_batch(y):
    y_true,y_pred = y
    return iou_metric_batch(y_true,y_pred)


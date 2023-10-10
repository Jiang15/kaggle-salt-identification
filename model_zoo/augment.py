import random
import numpy as np
import cv2
from PIL import Image,ImageFilter,ImageEnhance
class augmentor:
    def __init__(self):
        pass
    def fit(self,x):
        pass
    def fit_transform(self,x):
        self.fit(x)
        return self.transform(x)
    def transform(self,x,label=False):
        pass
    def detransform(self,x,label=False):
        pass
class aug_ori(augmentor):
    def transform(self,x,label=False):
        return x
    def detransform(self,x,label=False):
        return x
class aug_flip(augmentor):
    def __init__(self,t):
        self.type = t
    def transform(self,x,label=False):
        if len(x.shape) == 2:
            x = np.expand_dims(x,axis=2)
        x = cv2.flip(x,self.type)
        if len(x.shape) == 2:
            x = np.expand_dims(x,axis=2)
        return x
    def detransform(self,x,label=False):
        if len(x.shape) == 2:
            x = np.expand_dims(x,axis=2)
        x = cv2.flip(x,self.type)
        if len(x.shape) == 2:
            x = np.expand_dims(x,axis=2)
        return x
    
class aug_rorate(augmentor):
    def __init__(self,degree=None):
        self.degree = degree
    def fit(self,x):
        self.degree = (random.random()-0.5)*5
    def transform(self,x,label=False):
        (h, w) = x.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, self.degree, 1.0)
        return np.expand_dims(cv2.warpAffine(x, M, (w, h)),axis=2)
    def detransform(self,x,label=False):
        (h, w) = x.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -self.degree, 1.0)
        return np.expand_dims(cv2.warpAffine(x, M, (w, h)),axis=2)
    
        
class aug_crop(augmentor):
    def __init__(self,l=None,r=None,u=None,d=None):
        self.l = l; self.r = r; self.u = u; self.d= d
    def fit(self,x):
        self.l = random.randint(0,30)
        self.r = random.randint(0,30)
        self.u = random.randint(0,30)
        self.d = random.randint(0,30)
    def transform(self,x,label=False):
        h,w = x.shape[0:2]
        new_x = x[self.u:h-self.d,self.l:w-self.r,:]
        new_x = cv2.resize(new_x,(w,h))
        return np.expand_dims(new_x,axis=2)
    def detransform(self,x,label=False):
        h,w = x.shape[0:2]
        new_x = cv2.resize(new_x,(w-self.l-self.r,h-self.u-self.d))
        ret_x = np.zeros_like(x)
        ret_x[self.u:h-self.d,self.l:w-self.r,:] = new_x
        return np.expand_dims(ret_x,axis=2)
class aug_extend(augmentor):
    def __init__(self,l=None,r=None,u=None,d=None):
        self.l = l; self.r = r; self.u = u; self.d= d
    def fit(self,x):
        self.l = random.randint(0,30)
        self.r = random.randint(0,30)
        self.u = random.randint(0,30)
        self.d = random.randint(0,30)
    def detransform(self,x,label=False):
        h,w = x.shape[0:2]
        new_x = cv2.resize(x,(w+self.l+self.r,h+self.u+self.d))
        new_x = new_x[self.u:self.u+h,self.l:self.l+w]
        return np.expand_dims(new_x,axis=2)
    def transform(self,x,label=False):
        h,w = x.shape[0:2]
        new_x = np.zeros((h+self.u+self.d,w+self.l+self.r,x.shape[2]))
        new_x[self.u:self.u+h,self.l:self.l+w] = x
        new_x = cv2.resize(new_x,(w,h))
        return np.expand_dims(new_x,axis=2)  
class aug_hist_balance(augmentor):
    def transform(self,x,label=False):
        if label: return x
        im = cv2.equalizeHist((x[:,:,0]*255).astype(np.uint8))
        return np.expand_dims(im/255,axis=2)
    def detransform(self,x,label=False):
        return x
class aug_sharp(augmentor):
    def transform(self,x,label=False):
        if label: return x
        im = Image.fromarray((x[:,:,0]*255).astype(np.uint8),mode='L')
        im = im.filter(ImageFilter.SHARPEN)
        return np.expand_dims(np.array(im)/255,axis=2)
    def detransform(self,x,label=False):
        return x
class aug_moresharp(augmentor):
    def transform(self,x,label=False):
        if label: return x
        im = Image.fromarray((x[:,:,0]*255).astype(np.uint8),mode='L')
        im = im.filter(ImageFilter.SHARPEN)
        im = im.filter(ImageFilter.SHARPEN)
        return np.expand_dims(np.array(im)/255,axis=2)
    def detransform(self,x,label=False):
        return x
class aug_flipsharp(augmentor):
    def transform(self,x,label=False):
        if label: return np.expand_dims(cv2.flip(x,1),axis=2)
        im = Image.fromarray((x[:,:,0]*255).astype(np.uint8),mode='L')
        im = im.filter(ImageFilter.SHARPEN)
        return np.expand_dims(cv2.flip(np.array(im)/255,1),axis=2)
    def detransform(self,x,label=False):
        return np.expand_dims(cv2.flip(x,1),axis=2)
    
class aug_blur(augmentor):
    def transform(self,x,label=False):
        if label: return x
        x = cv2.blur(x,(3,3))
        return np.expand_dims(x,axis=2)
    def detransform(self,x,label=False):
        return x
class aug_remove_mid(augmentor):
    def transform(self,x,label=False):
        x[33:-34,33:-34,:] = 0
        return x
def augmentation(X,y,y_other,augments=[[aug_ori(),1],[aug_flip(1),1],[aug_sharp(),1]#,[aug_remove_mid(),1]
                               ]):#,[aug_flipsharp(),1],,[aug_hist_balance(),1],[aug_extend(),1],
    #,[aug_moresharp(),1],[aug_extend(),1]
    X_ret = []; y_ret = []; y_other_ret = []
    sample_weight = []
    for augment,w in augments:
        for i in range(X.shape[0]):
            mx = X[i]; my = y[i]; othery = y_other[i]
            nx = augment.fit_transform(mx)
            ny = augment.transform(my,label=True)
            X_ret.append(nx)
            y_ret.append(ny)
            y_other_ret.append(othery)
            sample_weight.append(w)
    return np.array(X_ret),np.array(y_ret),np.array(y_other_ret),np.array(sample_weight)
def get_augmentation_data(X,augment_class,args={}):
    X_ret = []
    augments = []
    for i in range(X.shape[0]):
        augment = augment_class(**args)
        augments.append(augment)
        X_ret.append(augment.fit_transform(X[i]))
    return np.array(X_ret),augments
def get_deaugmentation_data(X,augments):
    X_ret = []
    for i in range(X.shape[0]):
        X_ret.append(augments[i].detransform(X[i]))

    return np.array(X_ret)

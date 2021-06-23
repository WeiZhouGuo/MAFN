
from sklearn import metrics
import numpy as np
import random as rd
import torch
import keras.backend as K
def tostring(config):
    keys_ = config.keys()
    keys_.sort()
    return '-'.join([key + '_' + str(config[key]) for key in keys_])
def one_mode(train_x):
    ones = torch.ones_like(train_x).cuda()
    p_g=torch.where(train_x !=0,ones,train_x)
    Affinity_matrix=torch.zeros([train_x.shape[1],train_x.shape[1]]).cuda()
    for i in p_g:
        j=i.unsqueeze(0)
        A=j.t()*j
        diag=torch.diag(i)
        tmp=A-diag
        Affinity_matrix=torch.add(Affinity_matrix,tmp)
    Aff_g_g=Affinity_matrix[:400,:400]
    Aff_g_g = torch.where(Aff_g_g < 1, torch.zeros_like(Aff_g_g), Aff_g_g)
    Aff_g_g = torch.where(Aff_g_g >= 1, torch.ones_like(Aff_g_g), Aff_g_g)
    Aff_g_c=Affinity_matrix[:400,400:]
    Aff_g_c = torch.where(Aff_g_c < 1, torch.zeros_like(Aff_g_c), Aff_g_c)
    Aff_g_c = torch.where(Aff_g_c >= 1, torch.ones_like(Aff_g_c), Aff_g_c)
    Aff_c_g=Affinity_matrix[400:,:400]
    Aff_c_g = torch.where(Aff_c_g < 1, torch.zeros_like(Aff_c_g), Aff_c_g)
    Aff_c_g = torch.where(Aff_c_g >= 1, torch.ones_like(Aff_c_g), Aff_c_g)
    Aff_c_c=Affinity_matrix[400:,400:]
    Aff_c_c = torch.where(Aff_c_c < 2, torch.zeros_like(Aff_c_c), Aff_c_c)
    Aff_c_c = torch.where(Aff_c_c >= 2, torch.ones_like(Aff_c_c), Aff_c_c)
    Aff_gg_gc=torch.cat([Aff_g_g,Aff_g_c],dim=-1)
    Aff_cg_cc=torch.cat([Aff_c_g,Aff_c_c],dim=-1)
    Affinity_matrix=torch.cat([Aff_gg_gc,Aff_cg_cc],dim=0)
    return norm(Affinity_matrix)
def norm(adj):
    adj += torch.eye(adj.shape[0]).cuda()
    degree = adj.sum(1)
    degree = torch.diag(torch.pow(degree, -0.5))
    return torch.mm(torch.mm(degree,adj),degree)

def Pre(pre_socre,truth):
    pre_socre=np.reshape(pre_socre,[-1])
    truth=np.reshape(truth,[-1])
    sum=0
    for i,j in zip(pre_socre,truth):
        i=int(i+0.5)
        if i==j:
            sum+=1

    return sum/len(pre_socre)


def auc(pre,truth):

    pre = np.reshape(pre, [-1])
    truth = np.reshape(truth, [-1])
    pre,truth=zip(*sorted(zip(pre,truth),key=lambda x:x[0]))
    x=0
    y=0
    kkk=sum(truth)
    auc=0
    for idx in range(len(pre)):
        x_=sum(truth[:idx])/kkk
        y_=(idx-sum(truth[:idx]))/kkk
        auc+=((y_+y)*(x_-x))/2
        x=x_
        y=y_

    return auc

def lookup(pre,std):

    for i in range(len(std)):
        print(pre[i],std[i])
def Evaluation_index(output_list,pred,test_y_list):
    y_yuce = pred
    test_y_l = test_y_list
    recall_score_test = metrics.recall_score(test_y_l, y_yuce, pos_label=0)
    acc_score_test = metrics.accuracy_score(test_y_l, y_yuce)
    precision_test = metrics.precision_score(test_y_l, y_yuce, pos_label=0)
    F1_score_test = metrics.f1_score(test_y_l, y_yuce, pos_label=0)
    test_auc = metrics.roc_auc_score(test_y_l, output_list)
    mcc,sn=matthews_correlation(test_y_l,y_yuce)
    return acc_score_test,recall_score_test,precision_test,F1_score_test,test_auc,mcc,sn
def matthews_correlation(y_true, y_pred):
    tn, fp, fn, tp=metrics.confusion_matrix(y_true, y_pred).ravel()
    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    sn=tp/(tp+fn)
    np.seterr(divide='ignore', invalid='ignore')
    return numerator / (denominator ),sn

if __name__=='__main__':
    pre=np.array([rd.random() if rd.random()>i else rd.random() for i in range(1000)])

    truth=np.array([0 if rd.random()<0.5 else 1 for _ in range(1000)])
    print(auc(pre,truth))

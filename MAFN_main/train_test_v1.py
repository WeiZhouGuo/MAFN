from loss import My_loss
from model import Net
from sklearn import metrics
from functions import Evaluation_index,one_mode,Pre
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
import torch.utils.data as Data
from keras.utils.np_utils import to_categorical

config = {'input_size':625,
          'E_node':625,
          'dim_feature':[400,200,25],
          'lr': 0.0001,
          'dropout':0.5 ,
          'dropout_attention':0.4,
          'beta1':0.9,
          'beta2':0.99,
          'lambda_reg':0.00003,
        'weight_decay':0.0003
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.FloatTensor
setup_seed(5)
# Unpacking the data
path = '../data/'
train_index = np.load(path + 'train_index.npy',encoding='latin1', allow_pickle=True)
test_index = np.load(path + 'test_index.npy',encoding='latin1', allow_pickle=True)

label = get_label()
os_time,os_state = get_time_state()

data= load_data()

ori_os_time = []
ori_os_state = []
label_test, y_pred_test = [], []
for i in range(5):
    print('\n')
    print(i + 1, 'th fold ######')

    x_train, x_test = np.array(data)[train_index[i]], np.array(data)[test_index[i]]
    y_train, y_test = np.array(label)[train_index[i]], np.array(label)[test_index[i]]
    y_test_onehot = to_categorical(y_test)
    y_train_onehots = to_categorical(y_train)

    x_train=torch.tensor(x_train, dtype=torch.float32)
    x_test=torch.tensor(x_test, dtype=torch.float32)
    y_test_onehot=torch.tensor(y_test_onehot, dtype=torch.float32)
    y_train_onehots=torch.tensor(y_train_onehots, dtype=torch.float32)
    y_test_gene_os_time = np.array(os_time)[test_index[i]]
    y_test_gene_os_state = np.array(os_state)[test_index[i]]

    train_torc_dataset = Data.TensorDataset(x_train, y_train_onehots)
    train_loader = Data.DataLoader(
        dataset=train_torc_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = Net(config['input_size'], config['E_node'], config['dropout'], config['dropout_attention'],
                config['dim_feature']).to(device)
    model.apply(weights_init_uniform_rule)
    criterion = My_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],betas = (config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)


    model.train()
    for epoch in range(9):
        train_loss = 0.0
        train_label, y_pred = [], []
        for batch_idx, (train_x, train_y) in enumerate(train_loader):

            train_x, train_y = Variable(train_x).to(device), Variable(train_y).to(device)
            Affinity_matrix_train = one_mode(train_x[:, :config['dim_feature'][0] + config['dim_feature'][1]])
            optimizer.zero_grad()
            pred, _ = model(train_x, Affinity_matrix_train)
            loss_epoch = criterion(train_x,pred, train_y)
            loss_epoch.backward()
            train_loss += loss_epoch.data
            optimizer.step()


            train_label.extend(train_y.detach().cpu().numpy().tolist())
            y_pred.extend(pred.detach().cpu().numpy().tolist())


        train_loss /= len(train_loader.dataset)
        train_label=np.array(train_label)[:, 1]
        y_pred_echo=np.argmax(np.array(y_pred),1)
        predict_score_echo=np.array(y_pred)[:, 1]
        acc_score_train, recall_score_train, precision_train, F1_score_train, train_auc, mcc_train, sn_train = Evaluation_index(
            predict_score_echo, y_pred_echo,train_label)
        print(
            'Train Epoch: {}\t train_acc:{:.6f}\t  train_auc:{:.6f}\t  train_Loss: {:.6f}\t  '
            'recall_train: {:.4f}\t  precision_train: {:.4f}\t F1_train: {:.4f}'.format(epoch, acc_score_train,
                                                                                  train_auc,
                                                                                  train_loss,
                                                                                  recall_score_train,
                                                                                  precision_train,
                                                                                  F1_score_train))


    model.eval()
    test_loss = 0
    test_x = Variable(x_test).to(device)
    test_y = Variable(y_test_onehot).to(device)
    Affinity_matrix_test = one_mode(test_x[:, :config['dim_feature'][0] + config['dim_feature'][1]])
    output_test, _ = model(test_x, Affinity_matrix_test)

    test_loss_echo=criterion(test_x,output_test, test_y).data.item()
    test_loss += test_loss_echo
    label_test.extend(test_y.detach().cpu().numpy().tolist())
    y_pred_test.extend(output_test.detach().cpu().numpy().tolist())
    ori_os_time.extend(y_test_gene_os_time)
    ori_os_state.extend(y_test_gene_os_state)

    label_test_echo=np.array(test_y.detach().cpu().numpy().tolist())[:, 1]
    y_pred_test_echo=np.argmax(np.array(output_test.detach().cpu().numpy().tolist()), 1)
    predict_score_test_echo=np.array(output_test.detach().cpu().numpy().tolist())[:, 1]
    acc_test_echo, recall_test_echo, precision_test_echo, F1_score_test_echo, test_auc_echo, mcc_test_echo, sn_test_echo = Evaluation_index(
        predict_score_test_echo, y_pred_test_echo, label_test_echo)
    print(
        'Train Epoch: {}\t acc_test: {:.6f}\t test_auc:{:.6f}\t test_Loss: {:.6f}\t  '
        'recall_val: {:.4f}\t  precision_test: {:.4f}\t F1_test: {:.4f}'.format(epoch,
                                                                                acc_test_echo,
                                                                                test_auc_echo, test_loss,
                                                                                recall_test_echo,
                                                                                precision_test_echo,
                                                                                F1_score_test_echo))

test_loss /= 5
label_test_all = np.array(label_test)[:, 1]
y_pred_test_all = np.argmax(np.array(y_pred_test), 1)
predict_score_test_all = np.array(y_pred_test)[:, 1]
acc_score_test, recall_score_test, precision_test, F1_score_test, test_auc, mcc_test, sn_test = Evaluation_index(
    predict_score_test_all, y_pred_test_all, label_test_all)

print(
        'Train Epoch: {}\t acc_test: {:.6f}\t test_auc:{:.6f}\t test_Loss: {:.6f}\t  '
        'recall_val: {:.4f}\t  precision_test: {:.4f}\t F1_test: {:.4f},'.format(epoch,
                                                                              acc_score_test,
                                                                              test_auc, test_loss,
                                                                              recall_score_test,
                                                                              precision_test,
                                                                              F1_score_test))
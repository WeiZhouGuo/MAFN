from model import Net
from data_helper import data_helper
from functions import Evaluation_index,one_mode,Pre,setup_seed,weights_init_uniform_rule
import numpy as np
import torch
from torch.autograd import Variable

config = {'input_size':625,
          'E_node':625,
          'dim_feature':[400,200,25],
          'lr': 0.001,
          'dropout':0.5 ,
          'dropout_attention':0.4,
          'beta1':0.9,
          'beta2':0.99,
          'lambda_reg':0.00003,
        'weight_decay':0.0003
}
setup_seed(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.FloatTensor
d=data_helper()
train_loader = d.get_train()
val_loader=d.get_val_data()
acc_score_val=0
model=Net(config['input_size'],config['E_node'],config['dropout'],config['dropout_attention'],config['dim_feature']).to(device)


model.apply(weights_init_uniform_rule)
criterion=torch.nn.BCELoss(size_average=True, reduce=True)
weight_decay=0.00001
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                             weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)



max_acc_value=0

for epoch in range(10):
    model.train()
    # scheduler.step()
    train_loss = 0.0
    train_label, y_pred = [], []
    i=0
    for batch_idx, (train_x, train_y) in enumerate(train_loader):
        train_x, train_y = Variable(train_x).to(device), Variable(train_y).to(device)
        Affinity_matrix_train = one_mode(train_x[:, :config['dim_feature'][0] + config['dim_feature'][1]])
        optimizer.zero_grad()
        train_output ,_= model(train_x,Affinity_matrix_train)
        loss = criterion(train_output, train_y)
        loss.backward()
        train_loss += loss.data
        optimizer.step()

        train_label.extend(train_y.detach().cpu().numpy().tolist())
        y_pred.extend(train_output.detach().cpu().numpy().tolist())

    train_label = np.array(train_label)[:, 1]
    y_pred_echo = np.argmax(np.array(y_pred), 1)
    predict_score_echo = np.array(y_pred)[:, 1]
    acc_score_train, recall_score_train, precision_train, F1_score_train, train_auc, _, _ = Evaluation_index(
            predict_score_echo, y_pred_echo, train_label)

    train_loss /= (batch_idx+1)
    label_val, y_pred_val = [], []
    model.eval()
    val_loss = 0
    idx=0
    for  val_x, val_y in  val_loader:
        with torch.no_grad():
            val_x= Variable(val_x).to(device)
        val_y = Variable(val_y).to(device)
        Affinity_matrix_val = one_mode(val_x[:, :config['dim_feature'][0] + config['dim_feature'][1]])
        output_val,_ = model(val_x,Affinity_matrix_val)

        label_val.extend(val_y.detach().cpu().numpy().tolist())
        y_pred_val.extend(output_val.detach().cpu().numpy().tolist())
        val_loss += criterion(output_val, val_y).data

        idx=idx+1
    val_loss /= len(val_loader.dataset)
    label_val_all = np.array(label_val)[:, 1]
    y_pred_val_all = np.argmax(np.array(y_pred_val), 1)
    predict_score_val_all = np.array(y_pred_val)[:, 1]
    acc_score_val, recall_score_val, precision_val, F1_score_val, val_auc, mcc_val, sn_val = Evaluation_index(
        predict_score_val_all, y_pred_val_all, label_val_all)

    # if max_acc_value < acc_score_val:
    #     torch.save(model.state_dict(), './saved/model.pth')
    #     max_acc_value = acc_score_val
    print(
        'Train Epoch: {}\t train_acc:{:.4f}\t  train_auc:{:.4f}\t acc_val: {:.4f}\t val_auc:{:.4f}\t '
        'recall_val: {:.4f}\t acc_val: {:.4f}\t precision_val: {:.4f}\t F1_val: {:.4f}'.format(epoch, precision_train,
                                                                                               train_auc,
                                                                                               acc_score_val, val_auc,
                                                                                               recall_score_val,
                                                                                               acc_score_val,
                                                                                               precision_val,
                                                                                               F1_score_val))
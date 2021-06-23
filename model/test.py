from model import Net
from data_helper import data_helper
from functions import Evaluation_index,one_mode,Pre,setup_seed
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
}
setup_seed(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.FloatTensor
d=data_helper()
test_loader=d.get_val_data()
model=Net(config['input_size'],config['E_node'],config['dropout'],config['dropout_attention'],config['dim_feature']).to(device)
model.load_state_dict(torch.load("./saved/model.pth"))
label_test, y_pred_test = [], []
model.eval()
test_loss = 0
idx = 0
for test_x, test_y in test_loader:
    test_x = Variable(test_x).to(device)
    test_y = Variable(test_y).to(device)
    Affinity_matrix_test = one_mode(test_x[:, :config['dim_feature'][0] + config['dim_feature'][1]])
    output_test, _ = model(test_x, Affinity_matrix_test)

    label_test.extend(test_y.detach().cpu().numpy().tolist())
    y_pred_test.extend(output_test.detach().cpu().numpy().tolist())


    idx = idx + 1
test_loss /= len(test_loader.dataset)
label_test_all = np.array(label_test)[:, 1]
y_pred_test_all = np.argmax(np.array(y_pred_test), 1)
predict_score_test_all = np.array(y_pred_test)[:, 1]
acc_score_test, recall_score_test, precision_test, F1_score_test, test_auc, mcc_test, sn_test = Evaluation_index(
    predict_score_test_all, y_pred_test_all, label_test_all)

print(
    'test_auc:{:.6f}\t '
    'recall_test: {:.4f}\t acc_test: {:.4f}\t precision_test: {:.4f}\t F1_test: {:.4f}'.format(
                                                                                          test_auc,
                                                                                           recall_score_test,
                                                                                           acc_score_test,
                                                                                           precision_test,
                                                                                           F1_score_test))
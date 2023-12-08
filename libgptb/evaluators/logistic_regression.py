import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
from sklearn.metrics import f1_score,classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from libgptb.evaluators.base_evaluator import BaseEvaluator


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        print(input_dim)
        y = y.to(device)
        num_classes = len(y[0])#y.max().item() + 1
        print(f'num_classes {num_classes}')
        classifier = LogisticRegression(input_dim, int(num_classes)).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.Sigmoid()
        criterion = nn.BCELoss() #nn.NLLLoss() #

        best_val_micro = 0
        best_test_micro = 0
        best_test_macro = 0
        best_epoch = 0
        report = ''

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = (classifier(x[split['test']])).detach().cpu().numpy()
                    test_micro,test_macro = f1_loss(y_test,y_pred)
                    # test_macro = f1_score(y_test, pred_transformed,average='macro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = (classifier(x[split['valid']])).detach().cpu().numpy()
                    
                    val_micro, val_macro = f1_loss(y_val,y_pred)

                    if val_micro > best_val_micro:
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        best_test_macro = test_macro
                        best_epoch = epoch
                        # print('best_test_results')
                        y_pred = (classifier(x[split['test']])).detach().cpu().numpy()
                        # print(classification_report(y_test, y_pred))
                        pred_sorted = np.argsort(y_pred, axis=1)

                        # the true number of labels for each node
                        num_labels = np.sum(y_test, axis=1)
                        # we take the best k label predictions for all nodes, where k is the true number of labels
                        pred_reshaped = []
                        for pr, num in zip(pred_sorted, num_labels):
                            pred_reshaped.append(pr[-int(num):].tolist())

                        # convert back to binary vectors
                        pred_transformed = MultiLabelBinarizer(classes=range(num_classes)).fit_transform(pred_reshaped)
                        report = classification_report(y_test.astype(np.int64), pred_transformed)

                    pbar.set_postfix({'best test F1Mi': best_test_micro, 'F1Ma': best_test_macro})
                    pbar.update(self.test_interval)
        print(report)
        return {
            'micro_f1': best_test_micro,
            'macro_f1': best_test_macro,
            'report': report
        }
        
def f1_loss(y, predictions):
    #y = y.detach().cpu().numpy()
    #predictions = predictions.detach().cpu().numpy()
    number_of_labels = y.shape[1]
    y = y.astype(np.int64)
    # find the indices (labels) with the highest probabilities (ascending order)
    pred_sorted = np.argsort(predictions, axis=1)
    # the true number of labels for each node
    num_labels = np.sum(y, axis=1)
    # we take the best k label predictions for all nodes, where k is the true number of labels
    pred_reshaped = []
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-int(num):].tolist())

    # convert back to binary vectors
    pred_transformed = MultiLabelBinarizer(classes=range(number_of_labels)).fit_transform(pred_reshaped)
    # print(f'pred_transformed {pred_transformed[0]}')
    f1_micro = f1_score(y, pred_transformed, average='micro')
    f1_macro = f1_score(y, pred_transformed, average='macro')
    return f1_micro, f1_macro
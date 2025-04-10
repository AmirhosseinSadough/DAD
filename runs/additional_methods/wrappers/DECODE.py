import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
print(torch.cuda.is_available())

class DECODE():
    def __init__(self, lr, mom_score, win_size):
        self.lr = lr
        self.mom_score = mom_score
        self.win_size = win_size

    def fit(self, X, y=None):

        X                   = torch.tensor(X, dtype=torch.float).to(device)
        R                   = torch.eye(X.shape[1], dtype=torch.float).to(device)
        NR                  = torch.zeros(X.shape[0]).to(device)
        self.p              = X.shape[1]

        for i in range(self.win_size,X.shape[0],1):
            XD              = F.linear(X[i-self.win_size:i+1,:], R)
            XD              = (XD.T @ XD) / len(XD)
            XD              -= torch.diag(torch.diag(XD))
            XD              = (((1.0) / ((self.p) - 1)) * XD @ R)
            XD              = self.lr * XD
            R               -= XD
            NR[i]           = torch.norm(R)



        NR                  = np.abs(np.gradient(NR.cpu())) 
        scores              = np.zeros(len(NR))
        scores[0]           = NR[0]
        for i in range(self.win_size,len(NR),1):
            scores[i]       = (1-self.mom_score)*scores[i-1] + self.mom_score*NR[i]       
        
        decision_scores = torch.nan_to_num(torch.tensor(scores), nan=0.0, posinf=0.0, neginf=0.0)

        self.decision_scores_ = decision_scores

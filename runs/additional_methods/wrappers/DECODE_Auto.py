import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
print(torch.cuda.is_available())


class DECODE_Auto():
    def __init__(self):
        self.mom_score = 0.2
        self.win_size = 0
        self.HPO_s = 0.075

    def fit(self, X, y=None):

        X                   = torch.tensor(X, dtype=torch.float).to(device)
        self.p              = X.shape[1]
        excluded_LRs = []

        while True:
            # warm-up
            LRs = [8e-1, 5e-1, 2e-1, 8e-2, 5e-2, 2e-2, 8e-3, 5e-3, 2e-3, 8e-4, 5e-4, 2e-4, 8e-5, 5e-5, 2e-5, 8e-6, 5e-6, 2e-6]
            LRs = [lr for lr in LRs if lr not in excluded_LRs]
            if not LRs:
                raise ValueError("All learning rates have been excluded. No suitable learning rate found.")
                        
            DXm = torch.zeros(len(LRs))
            len_hpo = int(self.HPO_s * X.shape[0])
            cnt = 0
            for lrr in LRs:
                DX = torch.zeros(len_hpo).to(device)
                RR                  = torch.eye(X.shape[1], dtype=torch.float).to(device)
                for z in range(len_hpo):
                    XD1              = F.linear(X[z-self.win_size:z+1,:], RR)
                    XD1              = (XD1.T @ XD1) / len(XD1)
                    XD1              -= torch.diag(torch.diag(XD1))
                    XD1              = (((1.0) / ((self.p) - 1)) * XD1 @ RR)
                    XD1              = lrr * XD1
                    RR               -= XD1
                    DXt             = F.linear(X[z-self.win_size:z+1,:], RR)
                    DX[z]           = torch.mean( (DXt.T @ DXt) / len(DXt) )
                DXm[cnt]            = torch.mean(DX)
                cnt                 = cnt + 1
            DXm[torch.isnan(DXm)] = 1.0
            DXm_list = DXm.cpu().tolist() 

            min_val = min(DXm_list)
            significant_values = [x for x in DXm_list if x <= (min_val + 0.025*min_val)]

            significant_values.sort()

            if len(significant_values) == 0:
                original_index = torch.median(DXm, dim=0).indices.item()
            elif len(significant_values) == 1:
                last_value = significant_values[-1] 
                original_index = DXm_list.index(last_value)
            elif len(significant_values) == 2:
                last_value = significant_values[-1] 
                original_index = DXm_list.index(last_value)
            elif len(significant_values) == 3:
                last_value = significant_values[-2] 
                original_index = DXm_list.index(last_value)                
            else:
                last_value = significant_values[-3]  
                original_index = DXm_list.index(last_value)

            LR_n = LRs[original_index]
            self.lr = LR_n

            X                   = torch.tensor(X, dtype=torch.float).to(device)
            length_X            = X.shape[0]
            Xn                   = torch.concat([X,X] , dim=0)
            R                   = torch.eye(Xn.shape[1], dtype=torch.float).to(device)
            NR                  = torch.zeros(Xn.shape[0]).to(device)
            R_curve             = torch.zeros(Xn.shape[0], Xn.shape[1], Xn.shape[1]).to(device)
            for i in range(self.win_size,Xn.shape[0],1):
                XD              = F.linear(Xn[i-self.win_size:i+1,:], R)
                XD              = (XD.T @ XD) / len(XD)
                XD              -= torch.diag(torch.diag(XD))
                XD              = (((1.0) / ((self.p) - 1)) * XD @ R)
                XD              = self.lr * XD
                R               -= XD
                NR[i]           = torch.norm(R)
                R_curve[i]      = R

            NR                  = np.abs(np.gradient(NR.cpu())) 
            scores              = np.zeros(len(NR))
            scores[0]           = NR[0]
            for i in range(self.win_size,len(NR),1):
                scores[i]       = (1-self.mom_score)*scores[i-1] + self.mom_score*NR[i]       
            decision_scores_pre = torch.nan_to_num(torch.tensor(scores), nan=0.0, posinf=0.0, neginf=0.0)
            
            decision_scores = decision_scores_pre[length_X:2*length_X]

            if torch.max(decision_scores) == 0:
                index = LRs.index(LR_n)
                offsets = [-3, -2, -1, 1, 2, 3]
                adjacent_indices = [index + off for off in offsets if 0 <= index + off < len(LRs)]                
                adjacent_LRs = [LRs[i] for i in adjacent_indices]

                excluded_LRs.append(LR_n)
                for lr in adjacent_LRs:
                    if lr not in excluded_LRs:
                        excluded_LRs.append(lr)
                
                continue    
            else:
                break 
        
        self.decision_scores_ = decision_scores
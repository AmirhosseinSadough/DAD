import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DAD:
    """
    DAD: Decorrelation-based Anomaly Detection.

    DAD detects anomalies by tracking how well a learned rotation R
    decorrelates a sliding window of the input stream. At each step, the
    window is projected through R, its empirical covariance is computed,
    and the off-diagonal (or full) energy of that covariance gives the
    anomaly score — low when R keeps the data decorrelated, high when the
    correlation structure shifts in either direction. R is then updated by
    a gradient step that pushes it toward better decorrelation.

    Three modes
    ------------
    manual  * DAD(mode='manual', lr=...)
            Single streaming pass with a fixed learning rate from the start; R adapts online throughout.

    auto    * DAD(mode='auto', ...)
            Two-phase warm start:
                Phase 1 - gradient-based hyperparameter optimization on a warm-up slice to learn an initial (R_init, lr_init).
                Phase 2 - streaming pass over the full series, continuing to adapt R with lr = lr_reduction * lr_init.

    semi    * DAD(mode='semi', ...)
            Learns (R_init, lr_init) once on a clean training set via the same hyperparameter optimization procedure, then freezes R and scores new data in
            decision_function() with lr=0 (pure scoring, no adaptation).

    Shared options
    ---------------
    win_size  - sliding window length used for the covariance estimate.
    mom_score - exponential smoothing factor applied to raw scores.
    normalize - standardize inputs before fitting/scoring; since DAD operates on covariance structure, unscaled feature variances would dominate the decorrelation signal.
    """
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        # shared
        mode: str = "manual", 
        data_type = "streaming",    # streaming / tabular
        lr: float = 1e-3,
        mom_score: float = 0.25,
        win_size: int = 0,
        hpo_n: float = 0.01,
        hpo_lr = 5e-1,              # auto 5e-1, semi 1e-1
        hpo_loglr = 1.0e-6,         # auto 1e-6, semi 1e-5
        hpo_max_epochs: int = 40,
        hpo_patience: int = 5,
        hpo_max_lr: float = 0.8,
        lr_reduction: float = 0.1,
        normalize = True

    ):
        if mode not in ("manual", "auto", "semi"):
            raise ValueError(f"mode must be 'manual', 'auto', or 'semi', got '{mode}'")

        self.mode = mode
        self.mom_score = mom_score
        self.win_size = win_size

        # manual
        self.lr = lr

        # auto
        self.hpo_n = hpo_n
        self.hpo_lr = hpo_lr
        self.hpo_loglr = hpo_loglr
        self.hpo_max_epochs = hpo_max_epochs
        self.hpo_patience = hpo_patience
        self.hpo_max_lr = hpo_max_lr
        self.lr_reduction = lr_reduction
        self.normalize = normalize
        self.data_type = data_type
        self.lr_init = None
        self.R_init = None
        self.normalizer = StandardScaler()
        
    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _stream_pass(self, X_t, lr_val, R_init=None, compute_scores=False):
        T, d = X_t.shape
        R       = torch.eye(d, dtype=torch.float32, device=device) if R_init is None else R_init.clone()
        NR      = torch.zeros(T, device=device) if compute_scores else None

        for i in range(self.win_size, T):
            XD = F.linear(X_t[i - self.win_size : i + 1, :], R)

            XD1 = (XD.T @ XD) / len(XD)
            XD2 = XD1 - torch.diag(torch.diag(XD1))
            XD2 = (1.0 / (d - 1)) * (XD2 @ R)
            R = R - lr_val * XD2 

            if compute_scores:
                sc = torch.norm(XD1)
                NR[i] = sc
        return R, NR.cpu().numpy() if compute_scores else None


    def _apply_momentum(self, NR_np):
        scores = np.zeros(len(NR_np))
        scores[0] = NR_np[0]
        for i in range(self.win_size, len(NR_np)):
            scores[i] = (1 - self.mom_score) * scores[i - 1] + self.mom_score * NR_np[i]
        return scores

    def _finalize_scores(self, NR_np):
        scores = self._apply_momentum(NR_np)
        decision_scores = torch.nan_to_num(torch.tensor(scores), nan=0.0, posinf=0.0, neginf=0.0)
        self.decision_scores_ = decision_scores.numpy()

    # ------------------------------------------------------------------
    # learning rate via gradient-based HPO
    # ------------------------------------------------------------------ 
    def _search_lr(self, X_warmup_t):
        """Phase 1: gradient-based search for the best streaming LR on the warm-up slice."""
        n = X_warmup_t.shape[0]

        log_lr = torch.tensor(np.log(self.hpo_loglr), dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([log_lr], lr=self.hpo_lr)
        corr = (X_warmup_t.T @ X_warmup_t) / n
        corr = corr - torch.diag(torch.diag(corr))
        best_loss, best_lr_val, best_R, trigger = float("inf"), None, None, 0
        pbar = tqdm(range(self.hpo_max_epochs), desc=f"DAD ({self.mode}): HPO")
        for _ in pbar:
            optimizer.zero_grad()
            lr_val = self.hpo_max_lr * torch.sigmoid(log_lr)
            R, _ = self._stream_pass(X_warmup_t, lr_val)
            
            loss = torch.norm(corr @ R)
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.6f}", "LR": f"{lr_val.item():.6f}"})

            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                best_lr_val = lr_val.item()
                with torch.no_grad():
                    best_R = R.detach().clone()
                trigger = 0
            else:
                trigger += 1
            if trigger >= self.hpo_patience:
                break

        return best_lr_val, best_R
    # ------------------------------------------------------------------
    # fit  –  dispatches to the right mode
    # ------------------------------------------------------------------
    def fit(self, X, y=None):
        print(f"Fitting  DAD in {self.mode} mode ...")
        if self.mode == "manual":
            return self._fit_manual(X)
        elif self.mode == "auto":
            return self._fit_auto(X)
        else:
            return self._fit_semi(X)
    # ------------------------------------------------------------------
    # Manual mode
    # ------------------------------------------------------------------
    def _fit_manual(self, X):
        if self.normalize == True:
            self.normalizer = StandardScaler()
            X_t = torch.tensor(self.normalizer.fit_transform(np.asarray(X)), dtype=torch.float).to(device)
        else:
            X_t = torch.tensor(X, dtype=torch.float).to(device)

        _, NR_np = self._stream_pass(X_t, self.lr, compute_scores=True)
        self._finalize_scores(NR_np)
        return self
    # ------------------------------------------------------------------
    # Auto mode 
    # ------------------------------------------------------------------
    def _fit_auto(self, X):
        
        X_np = np.asarray(X)
        if self.normalize == True:
            X_np_mean = np.mean(X_np, axis=0)
            X_np_std = np.std(X_np, axis=0)
            X_np_std[X_np_std < 1e-6] = 1.0
            X_np = (X_np - X_np_mean) / X_np_std

        if self.data_type == "tabular":
            n_hpo = min(max(50, int(self.hpo_n * X_np.shape[0])), X_np.shape[0])
            warmup_idx = np.random.default_rng(seed=42).choice(X_np.shape[0], size=n_hpo, replace=False)
            X_warmup_np = X_np[warmup_idx]
        else:
            X_warmup_np = X_np[:self.hpo_n]

        if self.normalize == True:
            X_warmup_np_mean = np.mean(X_warmup_np, axis=0)
            X_warmup_np_std = np.std(X_warmup_np, axis=0)
            X_warmup_np_std[X_warmup_np_std < 1e-6] = 1.0
            X_warmup_t = torch.tensor((X_warmup_np - X_warmup_np_mean) / X_warmup_np_std, dtype=torch.float32, device=device)
            X_t = torch.tensor((X_np - X_warmup_np_mean)/X_warmup_np_std, dtype=torch.float32, device=device)
        else:
            X_warmup_t = torch.tensor(X_warmup_np, dtype=torch.float32, device=device)
            X_t = torch.tensor(X_np, dtype=torch.float32, device=device)

        # Phase 1 – HPO
        self.lr_init, self.R_init = self._search_lr(X_warmup_t)
        # Phase 2 – streaming
        with torch.no_grad():
            _, NR_np = self._stream_pass(X_t, self.lr_reduction * self.lr_init, R_init=self.R_init, compute_scores=True)
        self._finalize_scores(NR_np)

        return self
    # ------------------------------------------------------------------
    # Semi-supervised mode  
    # ------------------------------------------------------------------
    def _fit_semi(self, X):
        X_np = np.asarray(X)
        if self.normalize == True:
            X_t = torch.as_tensor(self.normalizer.fit_transform(X_np), dtype=torch.float32, device=device)
        else:
            X_t = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        self.lr_init, self.R_init = self._search_lr(X_t)
        return self        

    def decision_function(self, X):
        if self.R_init is None:
            raise ValueError("Model has not been fitted yet. Please call fit(x_train) first.")
        X_np = np.asarray(X)
        if self.normalize == True:
            X_t = torch.as_tensor(self.normalizer.transform(X_np), dtype=torch.float32, device=device)
        else:
            X_t = torch.as_tensor(X_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            _, NR_np = self._stream_pass(X_t, 0.0, self.R_init, compute_scores=True)
        decision_scores = torch.nan_to_num(torch.tensor(NR_np), nan=0.0, posinf=0.0, neginf=0.0)
        return decision_scores.numpy()       

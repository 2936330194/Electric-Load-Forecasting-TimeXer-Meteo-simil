import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Set seed
fix_seed = 2026
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Import project components
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cal_eval, append_probabilistic_eval
from utils.quantile import QuantileLoss

QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
N_QUANTILES = len(QUANTILES)
P50_IDX = QUANTILES.index(0.5)
P10_IDX = QUANTILES.index(0.1)
P90_IDX = QUANTILES.index(0.9)

class LSTMQuantileBaseline(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_len=96, n_quantiles=7):
        super().__init__()
        self.output_len = output_len
        self.n_quantiles = n_quantiles
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len * n_quantiles)
        
    def forward(self, x):
        # x: [B, seq_len, input_size]
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # [B, output_len * n_quantiles]
        return out.view(x.size(0), self.output_len, self.n_quantiles)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LSTM Baseline for Electricity Load Forecasting")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--train_epochs", type=int, default=50, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    
    args = parser.parse_args()
    
    # Overwrite default parameters to match baseline requirements
    args.task_name = "long_term_forecast"
    args.model_id = "HunanLoad_2024_672_96"
    args.model = "LSTM"
    args.data = "custom"
    args.root_path = os.path.join(project_root, "data")
    args.data_path = "湖南省电力负荷2024.csv"
    args.features = "S"
    args.target = "load"
    args.freq = "min"
    args.embed = "timeF"
    args.seq_len = 672
    args.label_len = 0
    args.pred_len = 96
    args.enc_in = 1
    args.c_out = 1
    args.num_workers = 0
    args.checkpoints = os.path.dirname(__file__)
    args.loss = "Quantile"
    args.lradj = "cosine"
    args.inverse_eval = True
    args.des = "Exp"
    args.itr = 1
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load data loaders
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    # Check if checkpoint already exists to skip training
    best_weight_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    if os.path.exists(best_weight_path):
        print("Checkpoint found. Skipping training phase.")
        args.is_training = 0

    checkpoint_path = os.path.join(args.checkpoints, 'lstm_checkpoint.pth')
    model = LSTMQuantileBaseline(
        input_size=1, hidden_size=128, num_layers=2,
        output_len=args.pred_len, n_quantiles=N_QUANTILES
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss(QUANTILES)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    if args.is_training:
        print("Training LSTM Quantile baseline...")
        print(f"Quantiles: {QUANTILES}")
        t0 = time.time()
        for epoch in range(args.train_epochs):
            model.train()
            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                # LSTM quantile output: [B, pred_len, n_quantiles]
                outputs = model(batch_x)
                batch_y_target = batch_y[:, -args.pred_len:, -1:] # [B, pred_len, 1]
                
                loss = criterion(outputs, batch_y_target)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                
            # Validate
            model.eval()
            vali_loss = []
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    outputs = model(batch_x)
                    batch_y_target = batch_y[:, -args.pred_len:, -1:]
                    loss = criterion(outputs, batch_y_target)
                    vali_loss.append(loss.item())
            
            train_loss_avg = np.mean(train_loss)
            vali_loss_avg = np.mean(vali_loss)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.6f} | Vali Loss: {vali_loss_avg:.6f}")
            
            early_stopping(vali_loss_avg, model, args.checkpoints)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            adjust_learning_rate(optimizer, epoch + 1, args)
            
        print(f"Training completed in {time.time() - t0:.1f}s")
    
    # Load best weights
    loaded_weight = False
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        print("Loaded best model.")
        loaded_weight = True
    else:
        # Fallback to checkpoint_path
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Loaded checkpoint model.")
            loaded_weight = True

    if not loaded_weight and not args.is_training:
        raise FileNotFoundError(
            f"Model checkpoint not found: {best_weight_path}. "
            "Set --is_training/default is_training to 1 to train LSTM first."
        )
            
    # Evaluation on Test set
    model.eval()
    preds_p50 = []
    trues = []
    quantile_preds_all = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)  # [B, pred_len, n_quantiles]
            
            p50_pred = outputs[:, :, P50_IDX:P50_IDX+1]  # [B, pred_len, 1]
            preds_p50.append(p50_pred.detach().cpu().numpy())
            quantile_preds_all.append(outputs.detach().cpu().numpy())
            trues.append(batch_y[:, -args.pred_len:, -1:].detach().cpu().numpy())
            
    preds_p50 = np.concatenate(preds_p50, axis=0)  # [N, pred_len, 1]
    trues = np.concatenate(trues, axis=0)  # [N, pred_len, 1]
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)  # [N, pred_len, n_quantiles]
    
    print(f"Test shape: preds={preds_p50.shape}, trues={trues.shape}, quantiles={quantile_preds_all.shape}")
    
    # Save standardized results
    np.save(os.path.join(args.checkpoints, 'pred.npy'), preds_p50)
    np.save(os.path.join(args.checkpoints, 'true.npy'), trues)
    np.save(os.path.join(args.checkpoints, 'quantile_preds.npy'), quantile_preds_all)
    
    # Inverse scale to physical load scale
    if args.inverse_eval and test_data.scale:
        shape = trues.shape
        preds_inv = test_data.inverse_transform(preds_p50.reshape(shape[0]*shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform(trues.reshape(shape[0]*shape[1], -1)).reshape(shape)
        
        # Inverse transform all quantiles
        q_shape = quantile_preds_all.shape  # [N, pred_len, n_quantiles]
        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi:qi+1]
            q_inv = test_data.inverse_transform(
                q_slice.reshape(q_shape[0]*q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]
        
        np.save(os.path.join(args.checkpoints, 'pred_inv.npy'), preds_inv)
        np.save(os.path.join(args.checkpoints, 'true_inv.npy'), trues_inv)
        np.save(os.path.join(args.checkpoints, 'quantile_preds_inv.npy'), quantile_inv)
    else:
        preds_inv = preds_p50
        trues_inv = trues

    origin_quantiles = quantile_inv if 'quantile_inv' in locals() else quantile_preds_all
    origin_eval_df = cal_eval(trues_inv, preds_inv)
    origin_eval_df = append_probabilistic_eval(origin_eval_df, trues_inv, origin_quantiles, QUANTILES)
    print("[origin Eval] metrics:")
    print(origin_eval_df)

if __name__ == "__main__":
    main()

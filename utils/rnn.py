import torch
import torch.nn as nn
import numpy as np
from utils.runnarx import run_narx

# Define a RNN simples
class ControlRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, activation='tanh', dropout_rate=0):
        super(ControlRNN, self).__init__()
        # recebe as quatro últimas posições e dá a próxima posição como saída
        self.input_size = input_size + 2
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                          nonlinearity=activation, dropout=dropout_rate, 
                          batch_first=True)
        # recebe a próxima posição [x, y] e dá os thetas como saída
        #self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, goal):
        goal_repeated = goal.unsqueeze(1).repeat(1, x.shape[1], 1)

        x = torch.cat([x, goal_repeated], dim=2)

        pos, _ = self.rnn(x)
        new_pos = pos[:, -1, :]  # pegamos só a última saída
        thetas = self.fc(new_pos)
        return thetas

def train_net(model, tloader, vloader, num_epochs, optimizer, lossFunc=nn.MSELoss(), delta=None, patience=None, verbose=2, alpha=1.0):
    train_losses = []
    test_losses = []
    best_train_score = None
    best_val_score = None
    for e in range(num_epochs):
        train_loss = 0.0 # total loss during single epoch training
        val_loss = 0.0
        model.train()
        for X_batch, goal_batch, y_batch, target_pos_batch in tloader:
            #X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            pred_thetas = model(X_batch, goal_batch)
            if not torch.isfinite(pred_thetas).all():
                print("🚨 Detected non-finite values in pred_thetas, skipping batch.")
                continue
            pred_pos = run_narx(pred_thetas)
            loss_theta = lossFunc(pred_thetas, y_batch)  # calculates the loss function result
            loss_goal = lossFunc(pred_pos, target_pos_batch)  # calculates the loss function result
            
            total_loss = loss_theta + alpha * loss_goal
            
            
            optimizer.zero_grad() # clears x.grad for every parameter x in the optimizer.
            total_loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            optimizer.step() # updates the value of x using the gradient x.grad

            train_loss += np.sqrt(total_loss.item())

        model.eval()
        with torch.no_grad():
            for X_batch, goal_batch, y_batch, target_pos_batch in vloader:
                #X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred_thetas = model(X_batch, goal_batch)
                pred_pos = run_narx(pred_thetas)
                loss_theta = lossFunc(pred_thetas, y_batch)
                loss_goal = lossFunc(pred_pos, target_pos_batch)
                total_loss = loss_theta + alpha * loss_goal
                val_loss += np.sqrt(total_loss.item())

            avg_train_loss = train_loss / len(tloader)
            avg_val_loss = val_loss / len(vloader)
            if(verbose >= 1):
                print(f'Epoch [{e + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_val_loss:.4f}')
        train_losses.append(avg_train_loss)
        test_losses.append(avg_val_loss)

        # Armazenamento do melhor modelo com base no score de validação
        if((best_val_score is None) or (best_val_score > avg_val_loss)):
                    best_val_score = avg_val_loss
                    best_e = e
                    best_model = model
                    if(verbose >= 2):
                        print('best_model updated')

        # Early stopping com base no score de treinamento (não foi usado mas enf)
        if((delta is not None) and (patience is not None)):
            if((best_train_score is None) or (best_train_score-avg_train_loss >= delta)):
                counter = 0
                best_train_score = avg_train_loss
            else:
                counter += 1
                if(counter>=patience):
                    if(verbose >= 2):
                        print("Early Stopping!")
                    break

    return best_model, best_e, (train_losses, test_losses)
import torch
import torch.nn as nn
import numpy as np

class ControlRNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=1, activation='tanh', dropout_rate=0):
        super(ControlRNN, self).__init__()
        # Expect 4 positions (4x2 = 8) + goal (2) = 10 total input size per timestep
        self.rnn = nn.RNN(input_size + 2, hidden_size, num_layers,
                          nonlinearity=activation, dropout=dropout_rate,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size, 5)  # 5 joint angles

    def forward(self, x, goal):
        # Repeat goal to match sequence length and concatenate to input
        goal_repeated = goal.unsqueeze(1).repeat(1, x.shape[1], 1)  # (batch, seq_len, 2)
        x = torch.cat([x, goal_repeated], dim=2)  # (batch, seq_len, input+goal)
        pos, _ = self.rnn(x)
        new_pos = pos[:, -1, :]  # only last output
        thetas = self.fc(new_pos)
        return thetas

    def train_net(model, tloader, vloader, num_epochs, optimizer, alpha=1.0, lossFunc=nn.MSELoss(), delta=None, patience=None, verbose=2):
        train_losses = []
        test_losses = []
        best_train_score = None
        best_val_score = None

        def forward_kinematics(thetas):
            # Dummy forward kinematics to estimate end-effector (you’ll need a real one here)
            # For now assume a toy linear mapping as a placeholder
            return thetas[:, :2]  # Example: just pick first 2 for simplicity

        for e in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            model.train()
            for X_batch, goal_batch, y_batch, target_pos_batch in tloader:
                pred_thetas = model(X_batch, goal_batch)
                pred_pos = forward_kinematics(pred_thetas)

                loss_theta = lossFunc(pred_thetas, y_batch)
                loss_goal = lossFunc(pred_pos, target_pos_batch)

                total_loss = loss_theta + alpha * loss_goal

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                train_loss += np.sqrt(total_loss.item())

            model.eval()
            with torch.no_grad():
                for X_batch, goal_batch, y_batch, target_pos_batch in vloader:
                    pred_thetas = model(X_batch, goal_batch)
                    pred_pos = forward_kinematics(pred_thetas)

                    loss_theta = lossFunc(pred_thetas, y_batch)
                    loss_goal = lossFunc(pred_pos, target_pos_batch)

                    total_loss = loss_theta + alpha * loss_goal
                    val_loss += np.sqrt(total_loss.item())

                avg_train_loss = train_loss / len(tloader)
                avg_val_loss = val_loss / len(vloader)

                if verbose >= 1:
                    print(f'Epoch [{e + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_val_loss:.4f}')
            train_losses.append(avg_train_loss)
            test_losses.append(avg_val_loss)

            if (best_val_score is None) or (best_val_score > avg_val_loss):
                best_val_score = avg_val_loss
                best_e = e
                best_model = model
                if verbose >= 2:
                    print('best_model updated')

            if (delta is not None) and (patience is not None):
                if (best_train_score is None) or (best_train_score - avg_train_loss >= delta):
                    counter = 0
                    best_train_score = avg_train_loss
                else:
                    counter += 1
                    if counter >= patience:
                        if verbose >= 2:
                            print("Early Stopping!")
                        break

        return best_model, best_e, (train_losses, test_losses)

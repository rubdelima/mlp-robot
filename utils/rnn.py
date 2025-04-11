import torch
import torch.nn as nn
import numpy as np

# Define a RNN simples
class ControlRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, activation='tanh', dropout_rate=0):
        super(ControlRNN, self).__init__()
        # [Hiperparâmetros]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = activation
        self.dropout_rate = dropout_rate

        # [Blocos da rede neural]
        # recebe as quatro últimas posições e dá a próxima posição como saída
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
        self.relu = nn.ReLU()
        # recebe a próxima posição [x, y] e dá os thetas como saída
        #self.dropout = nn.Dropout(p=drop_rate)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        pos, _ = self.lstm(x)
        pos = self.relu(pos[:, -1, :])
        pos = self.fc1(pos)
        pos = self.tanh(pos)
        pos = self.fc2(pos)
        pos = self.fc3(pos)
        return pos # q não é pos, é theta

def train_net(model, tloader, vloader, num_epochs, optimizer, lossFunc=nn.MSELoss(), delta=None, patience=None, verbose=2):
    train_losses = []
    test_losses = []
    best_train_score = None
    best_val_score = None
    for e in range(num_epochs):
        train_loss = 0.0 # total loss during single epoch training
        val_loss = 0.0
        model.train()
        for i, (X_batch, y_batch) in enumerate(tloader):
            #X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            pred = model(X_batch) # predictions based on batch X_batch
            loss = lossFunc(pred, y_batch)  # calculates the loss function result
            optimizer.zero_grad() # clears x.grad for every parameter x in the optimizer.
            loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            optimizer.step() # updates the value of x using the gradient x.grad

            train_loss += loss.item()
            l = np.sqrt(loss.item()) # rmse loss
            train_loss += l # value of loss?
            #print(f'Epoch [{e + 1}/{num_epochs}], Step [{i + 1}/{len(tloader)}], Loss: {l:.4f} ')

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in vloader:
                #X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                l = np.sqrt(lossFunc(pred, y_batch).item()) #rmse loss
                val_loss += l

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
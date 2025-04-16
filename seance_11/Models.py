from torch import nn






def mlp(nchar, n_class):
    return nn.Sequential(nn.Linear(nchar, 50),
                         nn.ReLU(),
                         nn.Linear(50, 30),
                         nn.ReLU(),
                         nn.Linear(30, n_class))

        
def cnn(nchar, char_dim, n_class):
    return nn.Sequential(nn.Embedding(nchar, char_dim),
                         nn.Conv1d(char_dim, 50, 3),
                         nn.ReLU(),
                         nn.Conv1d(50, 50, 3),
                         nn.ReLU(),
                         nn.AdaptiveMaxPool1d(1),
                         nn.Linear(50, n_class))

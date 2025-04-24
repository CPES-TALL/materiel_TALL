from torch import nn, cat, zeros, Tensor





def mlp(nchar, n_class):
    return nn.Sequential(nn.Linear(nchar, 50),
                         nn.ReLU(),
                         nn.Linear(50, 30),
                         nn.ReLU(),
                         nn.Linear(30, n_class))




trans = nn.Module()
trans.forward = lambda x: x.transpose(1,2) # simply overriding the forward method does the job, sweet!!!

flat = nn.Module()
flat.forward = lambda x: x[0] # RNN layers also return funky things


def cnn(nchar, char_dim, n_class):
    return nn.Sequential(nn.Embedding(nchar, char_dim),
                         trans,
                         nn.Conv1d(char_dim, 50, 3),
                         nn.ReLU(),
                         nn.Conv1d(50, 50, 3),
                         nn.ReLU(),
                         nn.AdaptiveMaxPool1d(1),
                         trans,
                         nn.Linear(50, n_class),
                         flat)


caps = nn.Module()
caps.forward = lambda x: x[0][:,-1] # RNN layers also return funky things

def rnn(nchar, char_dim, n_class):
    return nn.Sequential(nn.Embedding(nchar, char_dim),
                         nn.GRU(char_dim, 50, 1, batch_first=True),
                         caps,
                         nn.Linear(50, n_class))






# another way to create a more complex network
class RNN_LM(nn.Module):

    def __init__(self, nchar, char_dim, nlang, lang_dim, hidden_dim):
        nn.Module.__init__(self)

        self.hidden_dim = hidden_dim
        
        self.L = nn.Embedding(nlang, lang_dim) # for representing languages
        self.C = nn.Embedding(nchar, char_dim) # for representing characters
    
        self.gru = nn.GRUCell(char_dim + lang_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, nchar) # to predict the next character


    def forward(self, lng, char, hidden=None):
        #print(lng, char, self.C.weight.shape, Tensor([char]).long(), Tensor(lng).long())
        if hidden == None:
            hidden = zeros((1, self.hidden_dim))

        v = cat([self.L(Tensor([lng]).long()), self.C(Tensor([char]).long())], dim=1)
        v = self.gru(v, hidden)

        scores = self.linear2(self.relu(self.linear(v)))
        
        return scores, v



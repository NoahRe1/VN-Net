import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from models.ODE.function import create_regularization_fns,set_function
from models.ODE.block import set_block

class M2_MLP(nn.Module):
  def __init__(self, opt, dataset):
    super().__init__()
    self.opt = opt
    self.m21 = nn.Linear(opt.get('hidden_dim'), opt.get('hidden_dim'))
    self.m22 = nn.Linear(opt.get('hidden_dim'), dataset.num_classes)

  def forward(self, x):
    x = F.dropout(x, self.opt.get('dropout'), training=self.training)
    x = F.dropout(x + self.m21(torch.tanh(x)), self.opt.get('dropout'), training=self.training)  # tanh not relu to keep sign, with skip connection
    x = F.dropout(self.m22(torch.tanh(x)), self.opt.get('dropout'), training=self.training)

    return x

class GCN(nn.Module):
    r"""An implementation of the Node Adaptive Graph Convolution Layer.
    For details see: `"Adaptive Graph Convolutional Recurrent Network
    for Traffic Forecasting" <https://arxiv.org/abs/2007.02842>`_
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        K (int): Filter size :math:`K`.
        embedding_dimensions (int): Number of node embedding dimensions.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int, embedding_dimensions: int,**dgnn_kwargs):
        super(GCN, self).__init__()
        self.K = K
        self.opt = dgnn_kwargs
        K_n = (self.K+1 if self.opt.get('dyG')==True else self.K)
        self.weights_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, K_n, in_channels, out_channels))
        self.bias_pool = torch.nn.Parameter(torch.Tensor(embedding_dimensions, out_channels))
        
        glorot(self.weights_pool)
        zeros(self.bias_pool)
        
        if self.opt.get('dyG') == True:
            self.k=torch.nn.Linear(in_channels,embedding_dimensions)
            self.q=torch.nn.Linear(in_channels,embedding_dimensions)

    def forward(self, X: torch.FloatTensor, E_list: list) -> torch.FloatTensor:
        r"""Making a forward pass.
        Arg types:
            * **X** (PyTorch Float Tensor) - Node features.
            * **E** (PyTorch Float Tensor) - Node embeddings.
        Return types:
            * **X_G** (PyTorch Float Tensor) - Hidden state matrix for all nodes.
        """
        E,E_t=E_list
        number_of_nodes = E.shape[0]
        
        # 静态的部分
        if self.opt.get('Temb') == True:
            supports = F.softmax(F.relu(torch.matmul(E_t, E_t.transpose(2, 1))), dim=2) #B N N
            support_set = [torch.eye(number_of_nodes).unsqueeze(0).repeat(E_t.shape[0],1,1).to(supports.device), supports]
        else:
            supports = F.softmax(F.relu(torch.matmul(E, E.transpose(1, 0))), dim=1) #N N
            support_set = [torch.eye(number_of_nodes).to(supports.device), supports]

        for _ in range(2, self.K): # 用3..跑的话
            support = torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            support_set.append(support)

        if self.opt.get('dyG') == True:
            # 动态的部分
            k = self.k(X)
            q = self.q(X) 
            score = torch.einsum("BNC, BnC -> BNn", k, q).contiguous()  # B, N, N
            score_norm=F.softmax(F.relu(score),dim=2)
            support_set.append(score_norm)
        
        # 合
        supports = torch.stack(support_set, dim=1) ## B 'k' N N
        
        W = torch.einsum('nd,dkio->nkio', E, self.weights_pool) 
        bias = torch.matmul(E, self.bias_pool)
        if self.opt.get('Temb') == True:
            X_G = torch.einsum("bknm,bmc->bknc", supports, X) ### 算出supports可立即计算,即提到前面
        else:
            X_G = torch.einsum("knm,bmc->bknc", supports, X)
        X_G = X_G.permute(0, 2, 1, 3)
        X_G = torch.einsum('bnki,nkio->bno', X_G, W) + bias
        return X_G

class DGNN(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, node_num: int, maxview: int, embedding_dimensions: int,**model_kwargs):
        super(DGNN, self).__init__()
        opt = model_kwargs.get('dgnn')
        self.opt = opt
        self.T = opt.get('time')
        self.maxview = maxview
        self.num_classes = out_channels
        #self.num_features = in_channels
        self.num_nodes = node_num
        self.hidden_dim = opt.get('hidden_dim')
        ##new
        self.GCNencoder = GCN(in_channels,self.hidden_dim,self.maxview,embedding_dimensions,**self.opt)

        self.f = set_function(opt)
        block = set_block(opt)
        time_tensor = torch.tensor([0, self.T])
        self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

        #self.m1 = nn.Linear(self.num_features, opt.get('hidden_dim'))

        if self.opt.get('use_mlp'):
            self.m11 = nn.Linear(opt.get('hidden_dim'), opt.get('hidden_dim'))
            self.m12 = nn.Linear(opt.get('hidden_dim'), opt.get('hidden_dim'))
        
        if opt.get('fc_out'):
            self.fc = nn.Linear(opt.get('hidden_dim'), opt.get('hidden_dim'))

        if opt.get('m2_mlp'):
            self.m2 = M2_MLP(opt)
        else:
            self.m2 = nn.Linear(opt.get('hidden_dim'), self.num_classes)

        if self.opt.get('batch_norm'):
            self.bn_in = torch.nn.BatchNorm1d(opt.get('hidden_dim')*self.num_nodes)
            self.bn_out = torch.nn.BatchNorm1d(opt.get('hidden_dim')*self.num_nodes)

        self.odeblock = block(self.f, self.regularization_fns, opt, t=time_tensor)
        self.odeblock.odefunc.GNN_postXN = self.GNN_postXN
        self.odeblock.odefunc.GNN_m2 = self.m2
        self.trusted_mask = None

    def getNFE(self):
        return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0
        self.odeblock.reg_odefunc.odefunc.nfe = 0

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def __repr__(self):
        return self.__class__.__name__
    
    def compute_energy(self,x,edge_index):
        energy = self.propagate(edge_index, x=x,energy = True)
        return torch.mean(energy,dim=0).item()

    def message(self, x_i,x_j,energy):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        #[E,1]*[E,channel]
        if energy:
            return (torch.linalg.norm(x_j-x_i, dim=1)**2).unsqueeze(dim=1)

    def forward_XN(self, x, E_list,pos_encoding=None):
        ###forward XN
        x = self.GCNencoder(x, E_list)
        self.odeblock.set_x0(x)
        if self.opt.get('function')=='gread':
            if self.opt.get('beta_diag') == True:
                self.odeblock.odefunc.Beta = self.odeblock.odefunc.set_Beta()#对角b

        if self.trusted_mask is not None:
            if self.opt.get('nox0')==True:
                self.odeblock.set_x0(x*0)

        if self.training and self.odeblock.nreg > 0:
            z, self.reg_states = self.odeblock(x,E_list)
        else:
            z = self.odeblock(x,E_list)
        return z

    def GNN_postXN(self, z):
        if self.opt.get('augment')==True:
            z = torch.split(z, z.shape[1] // 2, dim=1)[0]
        # Activation.
        if self.opt.get('XN_activation')==True:
            z = F.relu(z)
        # fc from bottleneck
        if self.opt.get('fc_out')==True:
            z = self.fc(z)
            z = F.relu(z)
        # Dropout.
        z = F.dropout(z, self.opt.get('dropout'), training=self.training)
        return z

    def forward(self, X, E_list,pos_encoding=None):
        z = self.forward_XN(X,E_list,pos_encoding)
        z = self.GNN_postXN(z)
        # Decode each node embedding to get node label.
        z = self.m2(z)
        return z
    
class DEGCN(nn.Module):
    def __init__(self, node_num: int, in_channels: int,
                 out_channels: int, max_view: int, embed_dim: int, first_layer: bool=False, **model_kwargs):
        super(DEGCN, self).__init__()
        
        self.config = model_kwargs
        self.number_of_nodes = node_num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_view = max_view
        self.embedding_dimensions = embed_dim
        self.first_layer = first_layer
        self.node_num =node_num
        self._setup_layers()

        if self.first_layer:
            self.node_embeddings = nn.Parameter(torch.randn(node_num, embed_dim), requires_grad=True)
            if self.config.get('dgnn').get('Temb')==True:
                self.month_embedding = nn.Parameter(torch.randn(12, embed_dim), requires_grad=True)
                self.day_embedding = nn.Parameter(torch.randn(31, embed_dim), requires_grad=True)
                self.time_embedding = nn.Parameter(torch.randn(24, embed_dim), requires_grad=True)

    def _setup_layers(self):
        self._gate = DGNN(in_channels = self.in_channels + self.out_channels,
                            out_channels = 2*self.out_channels,
                            node_num=self.node_num,
                            maxview = self.max_view,
                            embedding_dimensions = self.embedding_dimensions,
                            **self.config)
                           
        self._update = DGNN(in_channels = self.in_channels + self.out_channels,
                              out_channels = self.out_channels,
                              node_num=self.node_num,
                              maxview = self.max_view,
                              embedding_dimensions = self.embedding_dimensions,
                              **self.config)

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device)
        return H

    def forward(self, X, H, **kwargs)  -> torch.FloatTensor:
        if self.first_layer:
            if self.config.get('dgnn').get('Temb')==True:
                node_embeddings_emb=self.node_embeddings.unsqueeze(0) # (1, N, 32)
                m=X[...,-3] # (B, N)
                m_emb = self.month_embedding[(m-1).type(torch.LongTensor)] # (B, N, 32)
                node_embeddings_emb = torch.add(node_embeddings_emb, m_emb)
                
                d=X[...,-2] # (B, N)
                d_emb = self.day_embedding[(d-1).type(torch.LongTensor)] # (B, N, 32)
                node_embeddings_emb = torch.add(node_embeddings_emb, d_emb)
                
                t=X[...,-1] # (B, N)
                t_emb = self.time_embedding[(t).type(torch.LongTensor)] # (B, N, 32)
                node_embeddings_emb = torch.add(node_embeddings_emb, t_emb) 

                E_list = [self.node_embeddings, node_embeddings_emb]
            else:
                E_list = [self.node_embeddings,None]
            X = X[..., :-3]
        else:
            E_list = kwargs['E_list']

        H = self._set_hidden_state(X, H) # (B, N, 32)    
        X_H = torch.cat((X, H), dim=-1)
        Z_R = torch.sigmoid(self._gate(X_H, E_list))
        Z, R = torch.split(Z_R, self.out_channels, dim=-1)
        C = torch.cat((X, Z*H), dim=-1)
        HC = torch.tanh(self._update(C, E_list))
        H = R*H + (1-R)*HC

        return H, {'E_list': E_list}
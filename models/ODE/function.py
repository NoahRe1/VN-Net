import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch.nn.init import uniform_, xavier_uniform_
from lib.utils_g import MaxNFEException
import six
import models.ODE.regularized_ODE_function as g_f
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros


class ODEFuncWrapper(nn.Module):
    def __init__(self, odefunc, E_list):
        super(ODEFuncWrapper, self).__init__()
        self.odefunc = odefunc
        self.E_list = E_list

    def forward(self, t, x):
        state = [x, self.E_list]
        return self.odefunc(t, state)


class FunctionNotDefined(Exception):
    pass


def set_function(opt):
    ode_str = opt.get("function")
    if ode_str == "gread":
        f = ODEFuncGread
    else:
        raise FunctionNotDefined
    return f


REGULARIZATION_FNS = {
    "kinetic_energy": g_f.quadratic_cost,
    "jacobian_norm2": g_f.jacobian_frobenius_regularization_fn,
    "total_deriv": g_f.total_derivative,
    "directional_penalty": g_f.directional_derivative,
}


class ODEFunc(MessagePassing):

    # currently requires in_features = out_features
    def __init__(self, opt):
        super(ODEFunc, self).__init__()
        self.opt = opt
        self.edge_index = None
        self.edge_weight = None
        self.attention_weights = None
        if opt.get("alpha_dim") == "sc":
            self.alpha_train = nn.Parameter(torch.tensor(0.0))
        elif opt.get("alpha_dim") == "vc":
            self.alpha_train = nn.Parameter(0.0 * torch.ones(1, opt.get("hidden_dim")))
        if opt.get("source_dim") == "sc":
            self.source_train = nn.Parameter(torch.tensor(0.0))
        elif opt.get("source_dim") == "vc":
            self.source_train = nn.Parameter(0.0 * torch.ones(1, opt.get("hidden_dim")))
        if opt.get("beta_dim") == "sc":
            self.beta_train = nn.Parameter(torch.tensor(0.0))
        elif opt.get("beta_dim") == "vc":
            self.beta_train = nn.Parameter(0.0 * torch.ones(1, opt.get("hidden_dim")))
        self.x0 = None
        self.nfe = 0
        self.alpha_sc = nn.Parameter(torch.ones(1))
        self.source_sc = nn.Parameter(torch.ones(1))

    def __repr__(self):
        return self.__class__.__name__


class ODEFuncGread(ODEFunc):

    # currently requires in_features = out_features
    def __init__(self, in_features, out_features, opt):
        super(ODEFuncGread, self).__init__(opt)
        self.in_features = in_features
        self.out_features = out_features

        self.reaction_tanh = False

        if opt.get("beta_diag") == True:
            self.b_W = nn.Parameter(torch.Tensor(in_features))
            self.reset_parameters()
        self.epoch = 0

    def reset_parameters(self):
        if self.opt.get("beta_diag") == True:
            uniform_(self.b_W, a=-1, b=1)

    def set_Beta(self, T=None):
        Beta = torch.diag(self.b_W)
        return Beta

    def sparse_multiply_new(self, X, E_list):
        E, E_t = E_list
        if E_t is not None:
            number_of_nodes = E_t.shape[1]
            supports = F.softmax(F.relu(torch.matmul(E_t, E_t.transpose(2, 1))), dim=2)

            ax = torch.einsum("bnm,bmc->bnc", supports, X)
        else:
            number_of_nodes = E.shape[0]
            supports = F.softmax(F.relu(torch.matmul(E, E.transpose(1, 0))), dim=1)

            ax = torch.einsum("nm,bmc->bnc", supports, X)

        return ax

    def forward(self, t, state):  # the t param is needed by the ODE solver.
        x, E_list = state
        if self.nfe > self.opt.get("max_nfe"):
            raise MaxNFEException
        self.nfe += 1
        if not self.opt.get("no_alpha_sigmoid"):
            alpha = torch.sigmoid(self.alpha_train)
            beta = torch.sigmoid(self.beta_train)
        else:
            alpha = self.alpha_train
            beta = self.beta_train

        """
        - `x` is equivalent $H$ in our paper.
        - `diffusion` is the diffusion term.
        """
        # ax = self.sparse_multiply(x)
        ax = self.sparse_multiply_new(x, E_list)
        diffusion = ax - x

        """
        - `reaction` is the reaction term.
        - We consider four `reaction_term` options
        - When `reaction_term` is bspm: GREAD-BS
        - When `reaction_term` is fisher: GREAD-F
        - When `reaction_term` is allen-cahn: GREAD-AC
        - When `reaction_term` is zeldovich: GREAD-Z
        - The `tanh` on reaction variable is optional, but we don't use in our experiments.
        """
        if self.opt.get("reaction_term") == "bspm":
            reaction = -self.sparse_multiply_new(diffusion, E_list)  # A(AX-X)
        elif self.opt.get("reaction_term") == "fisher":
            reaction = -(x - 1) * x
            if self.reaction_tanh == True:
                reaction = torch.tanh(reaction)
        elif self.opt.get("reaction_term") == "allen-cahn":
            reaction = -(x**2 - 1) * x
            if self.reaction_tanh == True:
                reaction = torch.tanh(reaction)
        elif self.opt.get("reaction_term") == "zeldovich":
            reaction = -(x**2 - x) * x
            if self.reaction_tanh == True:
                reaction = torch.tanh(reaction)
        elif self.opt.get("reaction_term") == "st":
            reaction = self.x0
        elif self.opt.get("reaction_term") == "fb":
            ax = -self.sparse_multiply_new(x, E_list)
            reaction = x - ax  # L = I - A
            if self.reaction_tanh == True:
                reaction = torch.tanh(reaction)
        elif self.opt.get("reaction_term") == "fb3":
            ax = -self.sparse_multiply_new(x, E_list)
            reaction = x - ax  # L = I - A
            if self.reaction_tanh == True:
                reaction = torch.tanh(reaction)
        else:
            reaction = 0.0

        """
        - `f` is in the reaction-diffusion form
        - $\mathbf{f}(\mathbf{H}(t)) := \frac{d \mathbf{H}(t)}{dt} = -\alpha\mathbf{L}\mathbf{H}(t) + \beta\mathbf{r}(\mathbf{H}(t), \mathbf{A})$
        - `beta_diag` is equivalent to $\beta$ with VC dimension
        - `self.Beta` is diagonal matrix intialized with gaussian distribution
        - Due to the diagonal matrix, it is same to the result of `beta*reaction` when `beta` is initialized with gaussian distribution.
        """
        if self.opt.get("reaction_term") == "diffusion":
            f = alpha * diffusion
        else:
            if self.opt.get("beta_diag") == False:
                if self.opt.get("reaction_term") == "fb":
                    f = alpha * diffusion + beta * reaction
                elif self.opt.get("reaction_term") == "fb3":
                    f = alpha * diffusion + beta * (reaction + x)
                else:
                    f = alpha * diffusion + beta * reaction
            elif self.opt.get("beta_diag") == True:
                f = alpha * diffusion + reaction @ self.Beta

        """
        - We do not use the `add_source` on GREAD
        """
        if self.opt.get("add_source"):
            f = f + self.source_train * self.x0
        return f


def create_regularization_fns(args):
    regularization_fns = []
    regularization_coeffs = []

    for arg_key, reg_fn in six.iteritems(REGULARIZATION_FNS):
        if args[arg_key] is not None:
            regularization_fns.append(reg_fn)
            regularization_coeffs.append(args[arg_key])

    regularization_fns = regularization_fns
    regularization_coeffs = regularization_coeffs
    return regularization_fns, regularization_coeffs

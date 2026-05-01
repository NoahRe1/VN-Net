import torch
from torch import nn
from torchdiffeq import odeint_adjoint, odeint
from functools import partial

from models.ODE.regularized_ODE_function import RegularizedODEfunc
from models.ODE.function import ODEFuncWrapper


class BlockNotDefined(Exception):
    pass


def set_block(opt):
    ode_str = opt.get("block")
    if ode_str == "constant":
        block = ConstantODEblock
    # elif ode_str == 'attention':
    # block = AttODEblock
    else:
        raise BlockNotDefined
    return block


class ODEblock(nn.Module):
    def __init__(self, odefunc, regularization_fns, opt, t):
        super(ODEblock, self).__init__()
        self.opt = opt
        self.t = t

        self.aug_dim = 2 if opt.get("augment") else 1
        self.odefunc = odefunc(
            self.aug_dim * opt.get("hidden_dim"),
            self.aug_dim * opt.get("hidden_dim"),
            opt,
        )
        # 11111111111111111
        self.nreg = len(regularization_fns)
        self.reg_odefunc = RegularizedODEfunc(self.odefunc, regularization_fns)

        if opt.get("adjoint"):
            self.train_integrator = odeint_adjoint
        else:
            self.train_integrator = odeint

        self.test_integrator = None
        self.set_tol()

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()
        self.reg_odefunc.odefunc.x0 = x0.clone().detach()

    def set_tol(self):
        self.atol = self.opt.get("tol_scale") * 1e-4
        self.rtol = self.opt.get("tol_scale") * 1e-3
        if self.opt.get("adjoint"):
            self.atol_adjoint = self.opt.get("tol_scale_adjoint") * 1e-4
            self.rtol_adjoint = self.opt.get("tol_scale_adjoint") * 1e-3

    def reset_tol(self):
        self.atol = self.opt.get("tol_scale") * 1e-4
        self.rtol = self.opt.get("tol_scale") * 1e-3
        self.atol_adjoint = self.opt.get("tol_scale_adjoint") * 1e-4
        self.rtol_adjoint = self.opt.get("tol_scale_adjoint") * 1e-3

    def set_time(self, time):
        self.t = torch.tensor([0, time]).to(self.device)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "( Time Interval "
            + str(self.t[0].item())
            + " -> "
            + str(self.t[1].item())
            + ")"
        )


class ConstantODEblock(ODEblock):
    def __init__(self, odefunc, regularization_fns, opt, t=torch.tensor([0, 1])):
        super(ConstantODEblock, self).__init__(odefunc, regularization_fns, opt, t)

        self.aug_dim = 2 if opt.get("augment") else 1
        self.odefunc = odefunc(
            self.aug_dim * opt.get("hidden_dim"),
            self.aug_dim * opt.get("hidden_dim"),
            opt,
        )
        #
        # if opt.get('data_norm') == 'rw':
        #   edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
        #                                        fill_value=opt.get('self_loop_weight'),
        #                                        num_nodes=data.num_nodes,
        #                                        dtype=data.x.dtype)
        # else:
        #   edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
        #                                               fill_value=opt.get('self_loop_weight'),
        #                                               num_nodes=data.num_nodes,
        #                                               dtype=data.x.dtype)
        # self.odefunc.edge_index = edge_index
        # self.odefunc.edge_weight = edge_weight
        # self.reg_odefunc.odefunc.edge_index, self.reg_odefunc.odefunc.edge_weight = self.odefunc.edge_index, self.odefunc.edge_weight

        if opt.get("method") == "symplectic_euler" or opt.get("method") == "leapfrog":
            from lib.odeint_geometric import odeint
        elif opt.get("adjoint"):
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        self.train_integrator = odeint
        self.test_integrator = odeint
        self.set_tol()

    def forward(self, x, E_list):
        t = self.t.type_as(x)

        integrator = self.train_integrator if self.training else self.test_integrator

        # Determine solver and adjoint options based on method
        solver_options = dict(step_size=self.opt.get("step_size"))
        adjoint_options = dict(step_size=self.opt.get("adjoint_step_size"))

        # Remove max_iters if using Euler or other solvers that do not support it
        method = self.opt.get("method")
        if method not in ["euler", "midpoint", "rk4"]:
            solver_options["max_iters"] = self.opt.get("max_iters")
            adjoint_options["max_iters"] = self.opt.get("max_iters")

        reg_states = tuple(torch.zeros(x.size(0)).to(x) for i in range(self.nreg))
        func = ODEFuncWrapper(
            self.reg_odefunc if self.training and self.nreg > 0 else self.odefunc,
            E_list,
        )
        state = x + reg_states if self.training and self.nreg > 0 else x

        if self.opt.get("adjoint") and self.training:
            state_dt = integrator(
                func,
                state,
                t,
                method=self.opt.get("method"),
                options=solver_options,
                adjoint_method=self.opt.get("adjoint_method"),
                adjoint_options=adjoint_options,
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint,
            )
        else:
            state_dt = integrator(
                func,
                state,
                t,
                method=self.opt.get("method"),
                options=solver_options,
                atol=self.atol,
                rtol=self.rtol,
            )

        if self.training and self.nreg > 0:
            z = state_dt[0][1]
            reg_states = tuple(st[1] for st in state_dt[1:])
            return z, reg_states
        else:
            self.odefunc.inter_step = state_dt
            z = state_dt[-1]
            return z

    def __repr__(self):
        return (
            self.__class__.__name__
            + "( Time Interval "
            + str(self.t[0].item())
            + " -> "
            + str(self.t[1].item())
            + ")"
        )

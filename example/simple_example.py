import math
import os

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam


def model(data):
    print("model")
    # define the hyperparameters that control the beta prior
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)
    # alpha0 = torch.nn.Parameter(torch.tensor(10.0))
    # beta0 = torch.nn.Parameter(torch.tensor(10.0))

    # sample f from the beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))

    print(alpha0, beta0, f)
    # loop over the observed data
    for i in range(len(data)):
        # observe datapoint i using the bernoulli likelihood
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])


def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    print("guide")
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


if __name__ == "__main__":
    # this is for running the notebook in our testing framework
    smoke_test = ('CI' in os.environ)
    n_steps = 2

    # enable validation (e.g. validate parameters of distributions)
    assert pyro.__version__.startswith('1.3.1')
    pyro.enable_validation(True)
    data = torch.tensor([0.0])
    for i in range(1):
        trace = poutine.trace(guide).get_trace(data)
        params = [trace.nodes[name]["value"].unconstrained() for name in trace.param_nodes]
        print(params, trace.nodes.keys())
        for k, v in trace.nodes.items():
            vv = v.get('value', None)  # type:torch.Tensor
            print(k)
            print(f"v = {vv} ({type(vv)}) {vv.unconstrained() if vv is not None and v['type'] == 'param' else None}")

            for ki, vi in v.items():
                print(f"\t{ki}: {vi}")

                if ki == 'fn' and v['type'] == "sample":
                    print("\tlog_prob", v['fn'].log_prob(torch.tensor(0.1)), v['fn'].log_prob(torch.tensor(1.0)))

    for i in range(1):
        trace = poutine.trace(model).get_trace(data)
        params = [trace.nodes[name]["value"].unconstrained() for name in trace.param_nodes]
        print(params, trace.nodes.keys())
        for k, v in trace.nodes.items():
            print(k)
            for ki, vi in v.items():
                print(f"\t{ki}: {vi}")

    pyro.clear_param_store()

    # create some data with 6 observed heads and 4 observed tails
    # data = torch.tensor([1.0, 1.0, 1.0, 0.0])
    # data = torch.tensor([1.0, 0.0, 0.0, 0.0])
    data = torch.tensor([1.0])
    # for _ in range(6):
    #     data.append(torch.tensor(1.0))
    # for _ in range(4):
    #     data.append(torch.tensor(0.0))

    # setup the optimizer
    adam_params = {"lr": 0.001, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # do gradient steps
    for step in range(n_steps):
        loss = svi.step(data)
        print(loss)
        if step % 100 == 0:
            print('.', end='')

    # grab the learned variational parameters
    alpha_q = pyro.param("alpha_q").item()
    beta_q = pyro.param("beta_q").item()

    # here we use some facts about the beta distribution
    # compute the inferred mean of the coin's fairness
    inferred_mean = alpha_q / (alpha_q + beta_q)
    # compute inferred standard deviation
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * math.sqrt(factor)

    print("\nbased on the data and our prior belief, the fairness " +
          "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

K = 2


def model(data):
    # Global variables.
    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    with pyro.plate("data", len(data)):
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)


def guide(data):
    pyro.param('scale', torch.tensor([0.5]))
    with pyro.plate('components', K):
        pyro.param('weights', torch.tensor([0.5]))
        pyro.param('locs', torch.tensor([0.5]))

if __name__ == "__main__":
    n_steps = 1000
    pyro.enable_validation(True)
    pyro.clear_param_store()
    pyro.set_rng_seed(0)

    n = 100
    head = 0.8
    data = torch.tensor([0., 1., 10., 11., 12.])
    adam_params = {"lr": 0.05, "betas": (0.9, 0.999)}
    optimizer = Adam(adam_params)

    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    loss = 0
    for step in range(n_steps):
        loss += svi.step(data)
        if (step + 1) % 200 == 0:
            print(f"[{step + 1}] {loss / (step + 1) / n:.2f}")

    map_estimates = global_guide(data)
    print(map_estimates)
    # latent = ['weights', 'locs', 'scale']
    # for lvar in latent:
    #     print(f"{lvar} {pyro.param(lvar).item()}")

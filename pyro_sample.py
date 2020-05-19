import pyro
from pyro.distributions import Bernoulli,Normal
from torch import tensor

if __name__ == "__main__":
    pyro.enable_validation(True)
    d0 = Bernoulli(tensor([[0.3, 0.5], [0.6, 0.5]]))
    d1 = d0.to_event(1)
    d2 = d0.to_event(2)
    print(d0, d0(), d0.batch_shape, d0.event_shape)
    print(d1, d1(), d1.batch_shape, d1.event_shape)
    print(d2, d2(), d2.batch_shape, d2.event_shape)

    x = pyro.sample("x", d0)
    print(x)
    x = pyro.sample("x", d1)
    print(x)
    x = pyro.sample("x", d2)
    print(x)

    x1 = pyro.sample("measurement", Normal(0, 0.75), obs=20.5)
    print(x1)

import torch
from torch.distributions import constraints, transform_to
import weakref
if __name__ == "__main__":
    unconstrained = torch.tensor([1.0, -2.0, 3.0, 4.0])
    print(unconstrained)
    constrained = transform_to(constraints.positive)(unconstrained)
    constrained.unconstrained = weakref.ref(unconstrained)
    print(constrained, constrained.unconstrained())

import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps
from pyro.ops.stats import quantile


# First we need some boilerplate to create a class and define a .model() method.
class Model1(ForecastingModel):
    # We then implement the .model() method. Since this is a generative model, it shouldn't
    # look at data; however it is convenient to see the shape of data we're supposed to
    # generate, so this inputs a zeros_like(data) tensor instead of the actual data.
    def model(self, zero_data, covariates):
        data_dim = zero_data.size(-1)  # Should be 1 in this univariate tutorial.
        feature_dim = covariates.size(-1)

        # The first part of the model is a probabilistic program to create a prediction.
        # We use the zero_data as a template for the shape of the prediction.
        bias = pyro.sample("bias", dist.Normal(0, 10).expand([data_dim]).to_event(1))
        weight = pyro.sample("weight", dist.Normal(0, 0.1).expand([feature_dim]).to_event(1))
        prediction = bias + (weight * covariates).sum(-1, keepdim=True)
        # The prediction should have the same shape as zero_data (duration, obs_dim),
        # but may have additional sample dimensions on the left.
        assert prediction.shape[-2:] == zero_data.shape

        # The next part of the model creates a likelihood or noise distribution.
        # Again we'll be Bayesian and write this as a probabilistic program with
        # priors over parameters.
        noise_scale = pyro.sample("noise_scale", dist.LogNormal(-5, 5).expand([1]).to_event(1))
        noise_dist = dist.Normal(0, noise_scale)

        # The final step is to call the .predict() method.
        self.predict(noise_dist, prediction)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    pyro.enable_validation(True)
    pyro.set_rng_seed(20200221)
    dataset = load_bart_od()
    print(dataset.keys())
    print(dataset["counts"].shape)
    print(" ".join(dataset["stations"]))

    T, O, D = dataset["counts"].shape
    data = dataset["counts"][:T // (24 * 7) * 24 * 7].reshape(T // (24 * 7), -1).sum(-1).log()
    data = data.unsqueeze(-1)
    plt.figure(figsize=(9, 3))
    plt.plot(data)
    plt.title("Total weekly ridership")
    plt.ylabel("log(# rides)")
    plt.xlabel("Week after 2011-01-01")
    plt.xlim(0, len(data))

    T0 = 0  # begining
    T2 = data.size(-2)  # end
    T1 = T2 - 52  # train/test split

    pyro.set_rng_seed(1)
    pyro.clear_param_store()
    time = torch.arange(float(T2)) / 365
    covariates = torch.stack([time], dim=-1)
    print(data.shape, covariates.shape)
    print(T1, T2)
    forecaster = Forecaster(Model1(), data[:T1], covariates[:T1], learning_rate=0.1)

    samples = forecaster(data[:T1], covariates, num_samples=1000)
    p10, p50, p90 = quantile(samples, (0.1, 0.5, 0.9)).squeeze(-1)
    crps = eval_crps(samples, data[T1:])
    print(samples.shape, p10.shape)

    plt.figure(figsize=(9, 3))
    plt.fill_between(torch.arange(T1, T2), p10, p90, color="red", alpha=0.3)
    plt.plot(torch.arange(T1, T2), p50, 'r-', label='forecast')
    plt.plot(data, 'k-', label='truth')
    plt.title("Total weekly ridership (CRPS = {:0.3g})".format(crps))
    plt.ylabel("log(# rides)")
    plt.xlabel("Week after 2011-01-01")
    plt.xlim(0, None)
    plt.legend(loc="best")
    plt.savefig("data/bart_plot_1.png")

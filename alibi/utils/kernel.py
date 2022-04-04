import numpy as np
from typing import Union, Optional
from alibi.utils.distance import squared_pairwise_distance


class GaussianRBF:
    def __init__(self, sigma: Optional[Union[float, np.ndarray]] = None, **kwargs) -> None:
        """
        Gaussian RBF kernel: :math:`k(x,y) = \\exp(-\\frac{||x-y||^2}{2\\sigma^2})`.
        A forward pass takes a batch of instances `x` of size `Nx x features` and `y` of size `Ny x features`
        and returns the kernel matrix of size `Nx x Ny`.

        Parameters
        ----------
        sigma
            Kernel bandwidth. Need not to be specified if being inferred or trained.
            Can pass multiple values to eval kernel with and then average.
        **kwargs
            Other arguments. Not used.
        """
        super().__init__()
        self.config = {'sigma': sigma}
        self.min_val, self.max_val = 0., 1.   # minimum and maximum values that the kernel can take.

        if sigma is None:
            self.log_sigma = np.empty(1, dtype=np.float32)
            self.init_required = True
        else:
            if not isinstance(sigma, np.ndarray):
                sigma = np.array(sigma)

            sigma = sigma.reshape(-1,).astype(np.float32)  # [Ns,]
            self.log_sigma = np.log(sigma)
            self.init_required = False

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.log_sigma)

    def __call__(self, x: np.ndarray, y: np.ndarray, infer_sigma: bool = False) -> np.ndarray:
        """
        Computes the kernel matrix between `x` and `y`.

        Parameters
        ----------
        x
            The first array of data instances.
        y
            The second array of data instances.
        infer_sigma
            Whether to infer `sigma` automatically. The `sigma` value is computed based on the median distance value
            between the instances from `x` and `y`.

        Returns
        -------
        Kernel matrix between `x` and `y` having the size of `Nx x Ny` where `Nx` is the number of instances in `x` \
        and `y` is the number of instances in `y`.
        """
        y = y.astype(x.dtype)
        x, y = x.reshape((x.shape[0], -1)), y.reshape((y.shape[0], -1))  # flatten
        dist = squared_pairwise_distance(x, y)  # [Nx, Ny]

        if infer_sigma or self.init_required:
            n = min(x.shape[0], y.shape[0])
            n = n if np.all(x[:n] == y[:n]) and x.shape == y.shape else 0
            n_median = n + (np.prod(dist.shape) - n) // 2 - 1
            sigma = np.expand_dims((.5 * np.sort(dist.reshape(-1))[n_median]) ** .5, axis=0)
            self.log_sigma = np.log(sigma)
            self.init_required = False

        gamma = np.array(1. / (2. * self.sigma ** 2), dtype=x.dtype)   # [Ns,]
        # TODO: do matrix multiplication after all?
        kernel_mat = np.exp(-np.concatenate([(g * dist)[None, :, :] for g in gamma], axis=0))  # [Ns, Nx, Ny]
        return np.mean(kernel_mat, axis=0)  # [Nx, Ny]


class GaussianRBFDistance:
    def __init__(self, sigma: Optional[Union[float, np.ndarray]] = None, **kwargs):
        """
        Gaussian RBF kernel dissimilarity/distance: :math:`k(x, y) = 1 - \\exp(-\\frac{||x-y||^2}{2\\sigma^2})`.
        A forward pass takes a batch of instances `x` of size `Nx x features` and `y` of size `Ny x features`
        and returns the kernel matrix of size `Nx x Ny`.

        Parameters
        ----------
        sigma
            See :py:meth:`alibi.utils.kernel.GaussianRBF.__init__`.
        **kwargs
            Other arguments. Not used.
        """
        super().__init__()
        self.kernel = GaussianRBF(sigma=sigma)

    def __call__(self, x: np.ndarray, y: np.ndarray, infer_sigma: bool = False) -> np.ndarray:
        kmatrix = self.kernel(x, y, infer_sigma)
        return self.kernel.max_val - kmatrix


class EuclideanDistance:
    def __init__(self, **kwargs) -> None:
        """
        Euclidean distance: :math:`k(x, y) = ||x-y||^2`. A forward pass takes a batch of instances `x` of
        size `Nx x features` and `y` of size `Ny x features` and returns the kernel matrix `Nx x Ny`.

        Parameters
        ----------
        **kwargs
            Other arguments. Not used.
        """
        self.min_val, self.max_val = 0., np.inf   # minimum and maximum values that the kernel can take.

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the kernel distance matrix between `x` and `y`.

        Parameters
        ----------
        x
            The first array of data instances.
        y
            The second array of data instances.

        Returns
        -------
        Kernel distance matrix between `x` and `y` having the size of `Nx x Ny`, where `Nx` is the number of \
        instances in `x` and `y` is the number of instances in `y`.
        """
        y = y.astype(x.dtype)
        x, y = x.reshape((x.shape[0], -1)), y.reshape((y.shape[0], -1))  # flatten
        dist = squared_pairwise_distance(x, y)  # [Nx, Ny]
        return dist
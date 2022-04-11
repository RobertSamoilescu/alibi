import numpy as np
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Dict, Callable, List, Sequence, Union
from sklearn.tree import DecisionTreeClassifier
from alibi.tests.utils import issorted


class Discretizer(metaclass=ABCMeta):
    def __init__(self, data: np.ndarray, numerical_features: List[int], feature_names: List[str]) -> None:
        """
        Initialize the discretizer.

        Parameters
        ----------
        data
            Data to discretize.
        numerical_features
            List of indices corresponding to the continuous feature columns. Only these features will be discretized.
        feature_names
            List with feature names.
        """
        self.to_discretize = numerical_features

        bins = self.bins(data)
        bins = [np.unique(x) for x in bins]

        self.feature_intervals = {}  # type: Dict[int, list]
        self.lambdas = {}            # type: Dict[int, Callable]

        for feature, ft_bins in zip(self.to_discretize, bins):
            # get nb of borders (nb of bins - 1) and the feature name
            n_bins = ft_bins.shape[0]
            name = feature_names[feature]

            # create names for bins of discretized features
            self.feature_intervals[feature] = ['%s <= %.2f' % (name, ft_bins[0])]
            for i in range(n_bins - 1):
                self.feature_intervals[feature].append('%.2f < %s <= %.2f' % (ft_bins[i], name, ft_bins[i + 1]))
            self.feature_intervals[feature].append('%s > %.2f' % (name, ft_bins[n_bins - 1]))
            self.lambdas[feature] = partial(self.get_bins, bins=ft_bins)

    @staticmethod
    def get_bins(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Discretizes the data in `x` using the bins in `bins`. This is achieved by searching for the index of
        each value in `x` into `qts`, which is assumed to be a 1-D sorted array.

        Parameters
        ----------
        x
            A `numpy` array of data to be discretized.
        bins
            A `numpy` array of bins. This should be a 1-D array sorted in ascending order.

        Returns
        -------
        A discretized data `numpy` array.
        """

        if len(bins.shape) != 1:
            raise ValueError("Expected 1D quantiles array!")
        if not issorted(bins):
            raise ValueError("Quantiles array should be sorted!")
        return np.searchsorted(bins, x)

    @abstractmethod
    def bins(self, data: np.ndarray) -> List[np.ndarray]:
        raise NotImplemented('Must implement `bins` method.')

    def discretize(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        data
            Data to discretize.

        Returns
        -------
        Discretized version of data with the same dimension.
        """

        data_disc = data.copy()
        for feature in self.lambdas:
            if len(data.shape) == 1:
                data_disc[feature] = int(self.lambdas[feature](data_disc[feature]))
            else:
                data_disc[:, feature] = self.lambdas[feature](data_disc[:, feature]).astype(int)

        return data_disc


class QuantilesDiscretizer(Discretizer):
    def __init__(self,
                 data: np.ndarray,
                 numerical_features: List[int],
                 feature_names: List[str],
                 qts: Sequence[Union[int, float]]) -> None:
        """
        Initialize the discretizer.

        Parameters
        ----------
        data
            Data to discretize.
        numerical_features
            List of indices corresponding to the continuous feature columns. Only these features will be discretized.
        feature_names
            List with feature names.
        qts
            Quantiles to be used to discretize the data.
        """
        self.qts = qts
        super().__init__(data=data, numerical_features=numerical_features, feature_names=feature_names)

    def bins(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Parameters
        ----------
        data
            Data to discretize.

        Returns
        -------
        List with bin values for each feature that is discretized.
        """
        bins = []
        for feature in self.to_discretize:
            qts = np.array(np.percentile(data[:, feature], self.qts))
            bins.append(qts)

        return bins


class QuartilesDiscretizer(QuantilesDiscretizer):
    def __init__(self, data: np.ndarray, numerical_features: List[int], feature_names: List[str]) -> None:
        """
        Initialize the discretizer.

        Parameters
        ----------
        data
            Data to discretize.
        numerical_features
            List of indices corresponding to the continuous feature columns. Only these features will be discretized.
        feature_names
            List with feature names.
        """
        qts = (25, 50, 75)
        super().__init__(data=data, numerical_features=numerical_features, feature_names=feature_names, qts=qts)


class DecilesDiscretizer(QuantilesDiscretizer):
    def __init__(self, data: np.ndarray, numerical_features: List[int], feature_names: List[str]) -> None:
        """
        Initialize the discretizer.

        Parameters
        ----------
        data
            Data to discretize.
        numerical_features
            List of indices corresponding to the continuous feature columns. Only these features will be discretized.
        feature_names
            List with feature names.
        """
        qts = (10, 20, 30, 40, 50, 60, 70, 80, 90)
        super().__init__(data=data, numerical_features=numerical_features, feature_names=feature_names, qts=qts)


class LinspaceDiscretizer(Discretizer):
    def __init__(self,
                 data: np.ndarray,
                 numerical_features: List[int],
                 feature_names: List[str],
                 nums: int = 10) -> None:
        """
        Initialize the discretizer.

        Parameters
        ----------
        data
            Data to discretize.
        numerical_features
            List of indices corresponding to the continuous feature columns. Only these features will be discretized.
        feature_names
            List with feature names.
        nums
            Number of equidistant bins to be used for discretization. Default is 10.
        """
        self.nums = nums
        super().__init__(data=data, numerical_features=numerical_features, feature_names=feature_names)

    def bins(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Parameters
        ----------
        data
            Data to discretize.

        Returns
        -------
        List with bin values for each feature that is discretized.
        """
        bins = []
        for feature in self.to_discretize:
            min_val, max_val = np.min(data[:, feature]), np.max(data[:, feature])
            linspace = np.linspace(start=min_val, stop=max_val, num=self.nums)
            bins.append(linspace)

        return bins


class EntropyDiscretizer(Discretizer):
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 numerical_features: List[int],
                 feature_names: List[str],
                 **kwargs) -> None:
        """
        Initialize the discretizer.

        Parameters
        ----------
        data
            Data to discretize.
        labels
            Data labels to be used for entropy-based discretization.
        numerical_features
            List of indices corresponding to the continuous feature columns. Only these features will be discretized.
        feature_names
            List with feature names.
        **kwargs
            Other arguments to be passed to the Decision Tree. The entropy-based discretization uses a Decision Tree
            splits for the bins. The criterion is always set to ``'entropy'``. If `max_depth` not specified, then
            the value will be set to 3.
        """
        self.labels = labels
        self.kwargs = kwargs
        self.kwargs.update({'criterion': 'entropy'})
        if 'max_depth' not in self.kwargs:
            self.kwargs.update({'max_depth': 3})

        super().__init__(data=data, numerical_features=numerical_features, feature_names=feature_names)

    def bins(self, data: np.ndarray):
        bins = []
        for feature in self.to_discretize:
            # Entropy splitting / at most 8 bins so max_depth=3
            dt = DecisionTreeClassifier(**self.kwargs)
            x = np.reshape(data[:, feature], (-1, 1))
            dt.fit(x, self.labels)
            ft_bins = dt.tree_.threshold[np.where(dt.tree_.children_left > -1)]

            if ft_bins.shape[0] == 0:
                ft_bins = np.array([np.median(data[:, feature])])
            else:
                ft_bins = np.sort(ft_bins)

            bins.append(ft_bins)

        return bins



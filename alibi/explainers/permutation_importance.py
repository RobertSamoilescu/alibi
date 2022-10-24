import copy
import inspect
import logging
import math
import numbers
import sys
from collections import defaultdict
from enum import Enum
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from tqdm import tqdm

from alibi.api.defaults import (DEFAULT_DATA_PERMUTATION_IMPORTANCE,
                                DEFAULT_META_PERMUTATION_IMPORTANCE)
from alibi.api.interfaces import Explainer, Explanation

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

logger = logging.getLogger(__name__)


class Method(str, Enum):
    """ Enumeration of supported method. """
    EXACT = 'exact'
    ESTIMATE = 'estimate'


class Kind(str, Enum):
    """ Enumeration of supported kind. """
    DIFFERENCE = 'difference'
    RATIO = 'ratio'


METRIC_FNS = {
    "loss": {
        # regression
        "mean_absolute_error": sklearn.metrics.mean_absolute_error,
        "mean_squared_error": sklearn.metrics.mean_squared_error,
    },
    "score": {
        # classification
        "accuracy": sklearn.metrics.accuracy_score,
        "precision": sklearn.metrics.precision_score,
        "recall": sklearn.metrics.recall_score,
        "f1": sklearn.metrics.f1_score,
        "roc_auc": sklearn.metrics.roc_auc_score,

        # regression
        "r2": sklearn.metrics.r2_score
    }
}
"""
Dictionary of supported string specified metrics

    - Loss functions
        - ``'mean_absolute_error`` - Mean absolute error regression loss. See `sklearn.metrics.mean_absolute_error`_ \
        for documentation.

        - ``'mean_squared_error'`` - Mean squared error regression loss. See `sklearn.metrics.mean_squared_error`_ \
        for documentation.

            .. _sklearn.metrics.mean_absolute_error:
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html#sklearn.metrics.mean_absolute_error

            .. _sklearn.metrics.mean_squared_error:
               https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error

    - Score functions
        - ``'accuracy'`` - Accuracy classification score. See `sklearn.metrics.accuracy_score` for documentation.

        - ``'precision'`` - Precision score. See `sklearn.metrics.precision_score` for documentation.

        - ``'recall'`` - Recall score. See `sklearn.metrics.recall_score` for documentation.

        - ``'f1_score'`` - F1 score. See `sklearn.metrics.f1_score` for documentation.

        - ``'roc_auc_score'`` - Area Under the Receiver Operating Characteristic Curve (ROC AUC) score. \
        See :py:meth:`alibi.utils.metrics.roc_auc_score` for documentation.

        - ``'r2_score'`` - :math:`R^2` (coefficient of determination) regression score. \
        See `sklearn.metrics.r2_score`_ for documentation.

            .. _sklearn.metrics.accuracy_score:
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

            .. _sklearn.metrics.precision_score:
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score

            .. _sklearn.metrics.recall_score:
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score

            .. _sklearn.metrics.f1_score:
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score

            .. _sklearn.metrics.r2_score:
                https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score

"""


class PermutationImportance(Explainer):
    """ Implementation of the permutation feature importance for tabular dataset. The method measure the importance
    of a feature as the relative increase/decrease in the loss/score function when the feature values are permuted.
    Supports black-box models.

    For details of the method see the papers:

     - https://link.springer.com/article/10.1023/A:1010933404324
     - https://arxiv.org/abs/1801.01489
    """

    def __init__(self,
                 predictor: Callable[[np.ndarray], np.ndarray],
                 feature_names: Optional[List[str]] = None,
                 verbose: bool = False):
        """
        Initialize the permutation feature importance.

        Parameters
        ----------
        predictor
            A prediction function which receives as input a `numpy` array of size `N x F`, and outputs a
            `numpy` array of size `N` (i.e. `(N, )`) or `N x T`, where `N` is the number of input instances,
            `F` is the number of features, and `T` is the number of targets. Note that the output shape must be
            compatible with the loss functions provided in the
            :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.
        feature_names
            A list of feature names used for displaying results.
        verbose
            Whether to print the progress of the explainer.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_PERMUTATION_IMPORTANCE))
        self.predictor = predictor
        self.feature_names = feature_names
        self.verbose = verbose

    def explain(self,  # type: ignore[override]
                X: np.ndarray,
                y: np.ndarray,
                loss_fns: Optional[
                    Union[
                        str,
                        List[str],
                        Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                        Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]
                    ]
                ] = None,
                score_fns: Optional[
                    Union[
                        str,
                        List[str],
                        Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                        Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]
                    ]
                ] = None,
                features: Optional[List[Union[int, Tuple[int, ...]]]] = None,
                method: Literal["estimate", "exact"] = "estimate",
                kind: Literal["ratio", "difference"] = "ratio",
                n_repeats: int = 50,
                sample_weight: Optional[np.ndarray] = None) -> Explanation:
        """
        Computes the permutation feature importance for each feature with respect to the given loss or score
        functions and the dataset `(X, y)`.

        Parameters
        ----------
        X
            A `N x F` input feature dataset used to calculate the permutation feature importance. This is typically the
            test dataset.
        y
            A `N` (i.e. `(N, )`) ground-truth labels array corresponding the input feature `X`.
        loss_fns
            A loss function or a dictionary of loss functions having as keys the names of the loss functions and as
            values the loss functions. Lower values are better. Note that the `predictor` output must be compatible
            with every loss function. Every loss function is expected to receive the following arguments:

             - `y_true` : ``np.ndarray`` -  a `numpy` array of ground-truth labels.

             - `y_pred` : ``np.ndarray`` - a `numpy` array of model predictions. This corresponds to the output of
             the model

             - `sample_weight`: ``Optional[np.ndarray]`` - a `numpy` array of sample weights.

        score_fns
            A score function or a dictionary of score functions having as keys the names of the score functions and as
            values the score functions. Higher values are better. As with the `loss_fns`, the `predictor` output
            must be compatible with every score function and the score function must have the same signature
            presented in the `loss_fns` parameter description.
        features
            An optional list of features or tuples of features for which to compute the permutation feature
            importance. If not provided, the permutation feature importance will be computed for every single features
            in the dataset. Some example of `features` would be: ``[0, 2]``, ``[0, 2, (0, 2)]``, ``[(0, 2)]``,
            where ``0`` and ``2`` correspond to column 0 and 2 in `X`, respectively.
        method
            The method to be used to compute the feature importance. If set to ``'exact'``, a "switch" operation is
            performed across all observed pairs, by excluding pairings that are actually observed in the original
            dataset. This operation is quadratic in the number of samples (`N x (N - 1)` samples) and thus can be
            computationally intensive. If set to ``'estimate'``, the dataset will be divided in half. The values of
            the first half containing the ground-truth labels the rest of the features (i.e. features that are left
            intact) is matched with the values of the second half of the permuted features, and the other way around.
            This method is computationally lighter and provides estimate error bars given by the standard deviation.
        kind
            Whether to report the importance as the loss/score ratio or the loss/score difference.
            Available values are: ``'ratio'`` | ``'difference'``.
        n_repeats
            Number of times to permute the features. Considered only when ``method='estimate'``.
        sample_weight
            Optional weight for each sample instance.

        Returns
        -------
        explanation
            An `Explanation` object containing the data and the metadata of the permutation feature importance.
            See usage at `Permutation feature importance examples`_ for details

            .. _Permutation feature importance examples:
                https://docs.seldon.io/projects/alibi/en/stable/methods/PermutationImportance.html
        """
        n_features = X.shape[1]

        if (loss_fns is None) and (score_fns is None):
            raise ValueError('At least one loss function or a score function must be provided.')

        # initialize loss and score functions
        loss_fns = self._init_metrics(metric_fns=loss_fns, type='loss')
        score_fns = self._init_metrics(metric_fns=score_fns, type='score')

        # set the `features_names` when the user did not provide the feature names
        if self.feature_names is None:
            self.feature_names = [f'f_{i}' for i in range(n_features)]

        # construct `feature_names` based on the `features`. If `features` is ``None``, then initialize
        # `features` with all single feature available in the dataset.
        if features:
            feature_names = [tuple([self.feature_names[f] for f in features])
                             if isinstance(features, tuple) else self.feature_names[features]
                             for features in features]
        else:
            feature_names = self.feature_names  # type: ignore[assignment]
            features = list(range(n_features))

        # unaltered model predictions
        y_hat = self.predictor(X)

        # compute original loss
        loss_orig = PermutationImportance._compute_metrics(metric_fns=loss_fns,
                                                           y=y,
                                                           y_hat=y_hat,
                                                           sample_weight=sample_weight)

        # compute original score
        score_orig = PermutationImportance._compute_metrics(metric_fns=score_fns,
                                                            y=y,
                                                            y_hat=y_hat,
                                                            sample_weight=sample_weight)

        # compute permutation feature importance for every feature
        # TODO: implement parallel version - future work as it can be done for ALE too
        individual_feature_importance = []

        for ifeatures in tqdm(features, disable=not self.verbose):
            individual_feature_importance.append(
                self._compute_permutation_importance(
                    X=X,
                    y=y,
                    loss_fns=loss_fns,
                    score_fns=score_fns,
                    method=method,
                    kind=kind,
                    n_repeats=n_repeats,
                    sample_weight=sample_weight,
                    features=ifeatures,
                    loss_orig=loss_orig,
                    score_orig=score_orig
                )
            )

        # update meta data params
        self.meta['params'].update(feature_names=feature_names,
                                   method=method,
                                   kind=kind,
                                   n_repeats=n_repeats,
                                   sample_weight=sample_weight)

        # build and return the explanation object
        return self._build_explanation(feature_names=feature_names,  # type: ignore[arg-type]
                                       individual_feature_importance=individual_feature_importance)

    def _init_metrics(self,
                      metric_fns: Optional[
                          Union[
                              str,
                              List[str],
                              Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                              Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]
                          ]
                      ],
                      type: Literal['loss', 'score']
                      ) -> Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]]:
        """
        Helper function to initialize the loss and score functions.

        Parameters
        ----------
        metric_fns
            See `loss_fns` or `score_fns` as defined in
            :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.
        type
            Metric function type. Supported types: ``'loss'`` | ``'score'``.

        Returns
        -------
        Initialized loss and score functions.
        """
        if metric_fns is None:
            return {}

        if callable(metric_fns):
            return {type: metric_fns}

        if isinstance(metric_fns, str):
            metric_fns = [metric_fns]

        if isinstance(metric_fns, list):
            dict_metric_fns = {}

            for metric_fn in metric_fns:
                if not isinstance(metric_fn, str):
                    raise ValueError(f'The {type} inside {type}_fns must be of type `str`.')

                if metric_fn not in METRIC_FNS[type]:
                    raise ValueError(f'Unknown {type} name. Received {metric_fn}. '
                                     f'Supported values are: {list(METRIC_FNS[type].keys())}')

                dict_metric_fns[metric_fn] = METRIC_FNS[type][metric_fn]
            return dict_metric_fns

        return metric_fns

    @staticmethod
    def _compute_metrics(metric_fns: Dict[str, Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]],
                         y: np.ndarray,
                         y_hat: np.ndarray,
                         sample_weight: Optional[np.ndarray] = None,
                         metrics: Optional[Dict[str, List[float]]] = None):
        """
        Helper function to compute multiple metrics.

        Parameters
        ----------
        metric_fns
            A dictionary of metric functions having as keys the names of the metric functions and as
            values the metric functions.
        y
            Ground truth targets.
        y_hat
            Predicted outcome as returned by the classifier.
        sample_weight
            Weight of each sample instance.
        metrics
            An optional dictionary of metrics, having as keys the name of the metrics and as value the evaluation of
            the metrics.

        Returns
        -------
        Dictionary having as keys the metric names and as values the evaluation of the metrics.
        """
        if metrics is None:
            metrics = defaultdict(list)

        for metric_name, metric_fn in metric_fns.items():
            metrics[metric_name].append(
                PermutationImportance._compute_metric(
                    metric_fn=metric_fn,
                    y=y,
                    y_hat=y_hat,
                    sample_weight=sample_weight
                )
            )
        return metrics

    @staticmethod
    def _compute_metric(metric_fn: Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float],
                        y: np.ndarray,
                        y_hat: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Helper function to compute a metric. It also checks if the metric function contains in its signature the
        arguments `y_true`, `y_pred` or `y_score`, and optionally `sample_weight`.

        Parameters
        ----------
        metric_fn
            Metric function to be used. Note that the loss/score function must be compatible with the
            `y_true`, `y_pred`, and optionally with `sample_weight`.
        y, y_hat, sample_weight
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_metrics`.

        Returns
        -------
        Evaluation of the metric.
        """
        args = inspect.getfullargspec(metric_fn).args

        if 'y_true' not in args:
            raise ValueError('The `scoring` function must have the argument `y_true` in its definition.')

        if ('y_pred' not in args) and ('y_score' not in args):
            raise ValueError('The `scoring` function must have the argument `y_pred` or `y_score` in its definition.')

        kwargs: Dict[str, Optional[np.ndarray]] = {
            'y_true': y,
            'y_pred' if 'y_pred' in args else 'y_score': y_hat
        }

        if 'sample_weight' not in args:
            # some metrics might not support `sample_weight` such as:
            # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html#sklearn.metrics.max_error
            if sample_weight is not None:
                logger.warning(f"The loss function '{metric_fn.__name__}' does not support argument `sample_weight`. "
                               f"Calling the method without `sample_weight`.")

        else:
            # include `sample_weight` int the `kwargs` if the metric supports it
            kwargs['sample_weight'] = sample_weight

        return metric_fn(**kwargs)  # type: ignore [call-arg]

    def _compute_permutation_importance(self,
                                        X: np.ndarray,
                                        y: np.ndarray,
                                        loss_fns: Dict[
                                            str,
                                            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                                        ],
                                        score_fns: Dict[
                                            str,
                                            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                                        ],
                                        method: Literal["estimate", "exact"],
                                        kind: Literal["difference", "ratio"],
                                        n_repeats: int,
                                        sample_weight: Optional[np.ndarray],
                                        features: Union[int, Tuple[int, ...]],
                                        loss_orig: Dict[str, float],
                                        score_orig: Dict[str, float]) -> Dict[str, Any]:

        """
        Helper function to compute the permutation feature importance for a given feature or tuple of features.

        Parameters
        ----------
        X, y, loss_fns, score_fns, method, kind, n_repeats, sample_weight
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.#
        features
            The feature or the tuple of features to compute the permutation feature importance for.
        loss_orig
            Original loss value when the features are left intact. The loss is computed on the original datasets.
        score_orig
            Original score value when the feature are left intact. The score is computed on the original dataset.

        Returns
        --------
        A dictionary having as keys the metric names and as values the permutation feature importance associated
        with the corresponding metrics.
        """
        if method == Method.EXACT:
            # computation of the exact statistic which is quadratic in the number of samples
            return self._compute_exact(X=X,
                                       y=y,
                                       loss_fns=loss_fns,
                                       score_fns=score_fns,
                                       kind=kind,
                                       sample_weight=sample_weight,
                                       features=features,
                                       loss_orig=loss_orig,
                                       score_orig=score_orig)

        # sample approximation
        return self._compute_estimate(X=X,
                                      y=y,
                                      loss_fns=loss_fns,
                                      score_fns=score_fns,
                                      kind=kind,
                                      n_repeats=n_repeats,
                                      sample_weight=sample_weight,
                                      features=features,
                                      loss_orig=loss_orig,
                                      score_orig=score_orig)

    def _compute_exact(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       loss_fns: Dict[
                            str,
                            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                       ],
                       score_fns: Dict[
                           str,
                           Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                       ],
                       kind: str,
                       sample_weight: Optional[np.ndarray],
                       features: Union[int, Tuple[int, ...]],
                       loss_orig: Dict[str, float],
                       score_orig: Dict[str, float]):
        """
        Helper function to compute the "exact" value of the permutation feature importance.

        Parameters
        ----------
        X, y, loss_fns, score_fns, kind, sample_weight, features, loss_orig, score_orig
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_permutation_importance`.  # noqa

        Returns
        -------
        A dictionary having as keys the metric names and as values the permutation feature importance associated
        with the corresponding metrics.
        """
        y_hat = []
        weights: Optional[List[np.ndarray]] = [] if sample_weight else None

        for i in range(len(X)):
            # create dataset
            X_tmp = np.tile(X[i:i+1], reps=(len(X) - 1, 1))
            X_tmp[:, features] = np.delete(arr=X[:, features], obj=i, axis=0)

            # compute predictions
            y_hat.append(self.predictor(X_tmp))

            # create sample weights if necessary
            if sample_weight is not None:
                weights.append(np.full(shape=(len(X_tmp),), fill_value=sample_weight[i]))  # type: ignore[union-attr]

        # concatenate all predictions and construct ground-truth array. At this point, the `y_pre` vector
        # should contain `N x (N - 1)` predictions, where `N` is the number of samples in `X`.
        y_hat = np.concatenate(y_hat, axis=0)
        y = np.tile(y.reshape(-1, 1), reps=(1, len(X) - 1)).reshape(-1)

        if weights is not None:
            weights = np.concatenate(weights, axis=0)

        # compute loss values for the altered dataset
        loss_permuted = PermutationImportance._compute_metrics(metric_fns=loss_fns,
                                                               y=y,
                                                               y_hat=y_hat,  # type: ignore[arg-type]
                                                               sample_weight=weights)  # type: ignore[arg-type]

        # compute score values for the altered dataset
        score_permuted = PermutationImportance._compute_metrics(metric_fns=score_fns,
                                                                y=y,
                                                                y_hat=y_hat,  # type: ignore[arg-type]
                                                                sample_weight=weights)  # type: ignore[arg-type]

        # compute feature importance for the loss functions
        loss_feature_importance = PermutationImportance._compute_importances(metric_fns=loss_fns,
                                                                             metric_orig=loss_orig,
                                                                             metric_permuted=loss_permuted,
                                                                             kind=kind,
                                                                             lower_is_better=True)

        # compute feature importance for the score functions
        score_feature_importance = PermutationImportance._compute_importances(metric_fns=score_fns,
                                                                              metric_orig=score_orig,
                                                                              metric_permuted=score_permuted,
                                                                              kind=kind,
                                                                              lower_is_better=False)

        return {**loss_feature_importance, **score_feature_importance}

    def _compute_estimate(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          loss_fns: Dict[
                              str,
                              Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                          ],
                          score_fns: Dict[
                              str,
                              Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                          ],
                          kind: str,
                          n_repeats: int,
                          sample_weight: Optional[np.ndarray],
                          features: Union[int, Tuple[int, ...]],
                          loss_orig: Dict[str, float],
                          score_orig: Dict[str, float]):
        """
        Helper function to compute the "estimate" mean, standard deviation and sample values of the permutation
        feature importance.

        Parameters
        ----------
        X, y, loss_fns, score_fns, kind, sample_weight, features, loss_orig, score_orig
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_permutation_importance`.  # noqa

        Returns
        -------
        A dictionary having as keys the metric names and as values the permutation feature importance associated
        with the corresponding metrics.
        """
        N = len(X)
        start, middle, end = 0, N // 2, N if N % 2 == 0 else N - 1
        fh, sh = np.s_[start:middle], np.s_[middle:end]

        loss_permuted: Dict[str, List[float]] = defaultdict(list)
        score_permuted: Dict[str, List[float]] = defaultdict(list)

        for i in range(n_repeats):
            # get random permutation. Note that this includes also the last element
            shuffled_indices = np.random.permutation(len(X))

            # shuffle the dataset
            X_tmp, y_tmp = X[shuffled_indices], y[shuffled_indices]
            sample_weight_tmp = None if (sample_weight is None) else sample_weight[shuffled_indices]

            # permute values from the first half into the second half and the other way around
            fvals_tmp = X_tmp[fh, features]
            X_tmp[fh, features] = X_tmp[sh, features]
            X_tmp[sh, features] = fvals_tmp

            # compute scores
            y_tmp_hat = self.predictor(X_tmp[:end])
            y_tmp = y_tmp[:end]
            weights = None if (sample_weight_tmp is None) else sample_weight_tmp[:end]

            # compute loss values for the altered dataset
            loss_permuted = PermutationImportance._compute_metrics(metric_fns=loss_fns,
                                                                   y=y_tmp,
                                                                   y_hat=y_tmp_hat,
                                                                   sample_weight=weights,
                                                                   metrics=loss_permuted)

            # compute score values for the altered dataset
            score_permuted = PermutationImportance._compute_metrics(metric_fns=score_fns,
                                                                    y=y_tmp,
                                                                    y_hat=y_tmp_hat,
                                                                    sample_weight=weights,
                                                                    metrics=score_permuted)

        # compute feature importance for the loss functions
        loss_feature_importance = PermutationImportance._compute_importances(metric_fns=loss_fns,
                                                                             metric_orig=loss_orig,
                                                                             metric_permuted=loss_permuted,
                                                                             kind=kind,
                                                                             lower_is_better=True)

        # compute feature importance for the score functions
        score_feature_importance = PermutationImportance._compute_importances(metric_fns=score_fns,
                                                                              metric_orig=score_orig,
                                                                              metric_permuted=score_permuted,
                                                                              kind=kind,
                                                                              lower_is_better=False)

        return {**loss_feature_importance, **score_feature_importance}

    @staticmethod
    def _compute_importances(metric_fns: Dict[
                                str,
                                Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], float]
                             ],
                             metric_orig: Dict[str, float],
                             metric_permuted: Dict[str, List[float]],
                             kind: str,
                             lower_is_better: bool) -> Dict[str, Any]:
        """
        Helper function to compute the feature importance as the metric ration or the metric difference
        based on the `kind` parameter and the `lower_is_better` flag for multiple metric functions.

        Parameters
        ----------
        metric_fns
            A dictionary of metric functions having as keys the names of the metric functions and as
            values the metric functions.
        metric_orig
            A dictionary having as keys the names of the metric functions and as values the metric evaluations when
            the feature values are left intact.
        metric_permuted
            A dictionary having as keys the names of the metric functions and as values a list of metric evaluations
            when the feature values are permuted.
        kind
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.
        lower_is_better
            Whether lower metric value is better.

        Returns
        -------
        A dictionary having as keys the names of the metric functions and as values the feature importance or
        a dictionary containing the mean and the standard deviation of the feature importance, and the samples used to
        compute the two statistics for the corresponding metrics.
        """
        feature_importance = {}

        for metric_name in metric_fns:
            importance_values = [
                PermutationImportance._compute_importance(
                    metric_orig=metric_orig[metric_name],
                    metric_permuted=metric_permuted_value,
                    kind=kind,
                    lower_is_better=lower_is_better
                ) for metric_permuted_value in metric_permuted[metric_name]
            ]

            if len(importance_values) > 1:
                feature_importance[metric_name] = {
                    "mean": np.mean(importance_values),
                    "std": np.std(importance_values),
                    "samples": np.array(importance_values),
                }
            else:
                feature_importance[metric_name] = importance_values[0].item()

        return feature_importance

    @staticmethod
    def _compute_importance(metric_orig: float, metric_permuted: float, kind: str, lower_is_better: bool):
        """
        Helper function to compute the feature importance as the metric ratio or the metric difference
        based on the `kind` parameter and `lower_is_better` flag.

        Parameters
        ----------
        metric_orig
            Metric value when the feature values are left intact.
        metric_permuted
            Metric value when the feature value are permuted.
        kind
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain`.
        lower_is_better
            See :py:meth:`alibi.explainers.permutation_importance.PermutationImportance._compute_importances.

        Returns
        -------
        Importance score.
        """
        if lower_is_better:
            return metric_permuted / metric_orig if kind == Kind.RATIO else metric_permuted - metric_orig

        return metric_orig / metric_permuted if kind == Kind.RATIO else metric_orig - metric_permuted

    def _build_explanation(self,
                           feature_names: List[Union[str, Tuple[str, ...]]],
                           individual_feature_importance: List[
                               Union[
                                   Dict[str, float],
                                   Dict[str, Dict[str, float]]
                               ]
                           ]) -> Explanation:
        """
        Helper method to build `Explanation` object.

        Parameters
        ----------
        feature_names
            List of names of the explained features.
        individual_feature_importance
            List of dictionary having as keys the names of the metric functions and as values the feature
            importance when ``kind='exact'`` or a dictionary containing the mean and the standard deviation of the
            feature importance, and the samples used to compute the two statistics when``kind='estimate'`` for
            the corresponding metrics.

        Returns
        -------
        `Explanation` object.
        """
        # list of metrics names
        metric_names = list(individual_feature_importance[0].keys())

        # list of lists of features importance, one list per loss function
        feature_importance: List[List[Union[float, Dict[str, float]]]] = []

        for metric_name in metric_names:
            feature_importance.append([])

            for i in range(len(feature_names)):
                feature_importance[-1].append(individual_feature_importance[i][metric_name])

        data = copy.deepcopy(DEFAULT_DATA_PERMUTATION_IMPORTANCE)
        data.update(
            feature_names=feature_names,
            metric_names=metric_names,
            feature_importance=feature_importance,
        )

        return Explanation(meta=copy.deepcopy(self.meta), data=data)

    def reset_predictor(self, predictor: Callable) -> None:
        """
        Resets the predictor function.

        Parameters
        ----------
        predictor
            New predictor function.
        """
        self.predictor = predictor


# No type check due to the generic explanation object
@no_type_check
def plot_permutation_importance(exp: Explanation,
                                features: Union[List[int], Literal['all']] = 'all',
                                metric_names: Union[List[Union[str, int]], Literal['all']] = 'all',
                                n_cols: int = 3,
                                sort: bool = True,
                                top_k: int = 10,
                                ax: Optional[Union['plt.Axes', np.ndarray]] = None,
                                bar_kw: Optional[dict] = None,
                                fig_kw: Optional[dict] = None) -> 'plt.Axes':
    """
    Plot permutation feature importance on `matplotlib` axes.

    Parameters
    ----------
    exp
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method.
    features
        A list of feature entries provided in `feature_names` argument  to the
        :py:meth:`alibi.explainers.permutation_importance.PermutationImportance.explain` method, or
        ``'all'`` to  plot all the explained features. For example, if  ``feature_names = ['temp', 'hum', 'windspeed']``
        and we want to plot the values only for the ``'temp'`` and ``'windspeed'``, then we would set
        ``features=[0, 2]``. Defaults to ``'all'``.
    metric_names
        A list of metric entries in the `exp.data['metrics']` to plot the permutation feature importance for,
        or ``'all'`` to plot the permutation feature importance for all metrics (i.e., loss and score functions).
        The ordering is given by the concatenation of the loss metrics followed by the score metrics.
    n_cols
        Number of columns to organize the resulting plot into.
    sort
        Boolean flag whether to sort the values in descending order.
    top_k
        Number of top k values to be displayed if the ``sort=True``. If not provided, then all values will be displayed.
    ax
        A `matplotlib` axes object or a `numpy` array of `matplotlib` axes to plot on.
    bar_kw
        Keyword arguments passed to the `matplotlib.pyplot.barh`_ function.
    fig_kw
        Keyword arguments passed to the `matplotlib.figure.set`_ function.

        .. _matplotlib.pyplot.barh:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.barh.html

        .. _matplotlib.figure.set:
            https://matplotlib.org/stable/api/figure_api.html

    Returns
    --------
    `plt.Axes` with the feature importance plot.
    """
    from matplotlib.gridspec import GridSpec

    # define figure arguments
    default_fig_kw = {'tight_layout': 'tight'}
    if fig_kw is None:
        fig_kw = {}

    fig_kw = {**default_fig_kw, **fig_kw}

    # initialize `features` and `metric_names` if set to ``'all'``
    n_features = len(exp.data['feature_names'])
    n_metric_names = len(exp.data['metric_names'])

    if features == 'all':
        features = list(range(n_features))

    if metric_names == 'all':
        metric_names = exp.data['metric_names']

    # `features` sanity checks
    for ifeature in features:
        if ifeature >= len(exp.data['feature_names']):
            raise ValueError(f"The `features` indices must be les thant the "
                             f"``len(feature_names) = {n_features}``. Received {ifeature}.")

    # construct vector of feature names to display importance for
    feature_names = [exp.data['feature_names'][i] for i in features]

    # `metric_names` sanity checks
    for i, imetric_name in enumerate(metric_names):
        if isinstance(imetric_name, str) and (imetric_name not in exp.data['metric_names']):
            raise ValueError(f"Unknown `metric_name`. Received {imetric_name}. "
                             f"Available values are: {exp.data['metric_names']}.")

        if isinstance(imetric_name, numbers.Integral):
            if imetric_name >= n_metric_names:
                raise IndexError(f"Loss name index out of range. Received {imetric_name}. "
                                 f"The number of `metric_names` is {n_metric_names}")

            # convert index to string
            metric_names[i] = exp.data['metric_names'][i]

    if ax is None:
        fix, ax = plt.subplots()

    if isinstance(ax, plt.Axes) and n_metric_names != 1:
        ax.set_axis_off()  # treat passed axis as a canvas for subplots
        fig = ax.figure
        n_cols = min(n_cols, n_metric_names)
        n_rows = math.ceil(n_metric_names / n_cols)

        axes = np.empty((n_rows, n_cols), dtype=np.object)
        axes_ravel = axes.ravel()
        gs = GridSpec(n_rows, n_cols)

        for i, spec in zip(range(n_metric_names), gs):
            axes_ravel[i] = fig.add_subplot(spec)

    else:  # array-like
        if isinstance(ax, plt.Axes):
            ax = np.array(ax)

        if ax.size < n_metric_names:
            raise ValueError(f"Expected ax to have {n_metric_names} axes, got {ax.size}")

        axes = np.atleast_2d(ax)
        axes_ravel = axes.ravel()
        fig = axes_ravel[0].figure

    for i in range(n_metric_names):
        ax = axes_ravel[i]
        metric_name = metric_names[i]

        # define bar plot data
        y_labels = feature_names
        y_labels = ['(' + ', '.join(y_label) + ')' if isinstance(y_label, tuple) else y_label for y_label in y_labels]

        if exp.meta['params']['method'] == Method.EXACT:
            width = [exp.data['feature_importance'][i][j] for j in features]
            xerr = None
        else:
            width = [exp.data['feature_importance'][i][j]['mean'] for j in features]
            xerr = [exp.data['feature_importance'][i][j]['std'] for j in features]

        if sort:
            sorted_indices = np.argsort(width)[::-1][:top_k]
            width = [width[j] for j in sorted_indices]
            y_labels = [y_labels[j] for j in sorted_indices]

            if exp.meta['params']['method'] == Method.ESTIMATE:
                xerr = [xerr[j] for j in sorted_indices]

        y = np.arange(len(width))
        default_bar_kw = {'align': 'center'}
        bar_kw = default_bar_kw if bar_kw is None else {**default_bar_kw, **bar_kw}

        ax.barh(y=y, width=width, xerr=xerr, **bar_kw)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Permutation feature importance')
        ax.set_title(metric_name)

    fig.set(**fig_kw)
    return axes

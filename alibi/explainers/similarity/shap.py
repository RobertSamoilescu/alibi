import copy
from typing import Any, Callable, Dict, List, Literal

import numpy as np

from alibi.api.defaults import DEFAULT_DATA_SHAP_SIM, DEFAULT_META_SHAP_SIM
from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.explainers.shap_wrappers import KernelShap
from alibi.explainers.similarity.metrics import cos, dot


class ShapSimilarity(Explainer, FitMixin):
    NUM_RETURNED = 10

    def __init__(self,
                 kshap: KernelShap,
                 sim_fn: Literal['grad_dot', 'grad_cos'] = 'grad_cos'):
        """
        Constructor

        Parameters
        ----------
        kshap
            Fitted `KernelShap` explainer.
        sim_fn
            Similarity function name. Supported: ``'grad_dot'`` | ``'grad_cos'``.
        """
        super().__init__(meta=copy.deepcopy(DEFAULT_META_SHAP_SIM))
        self.meta['params'].update(sim_fn=sim_fn)

        sim_fn_opts: Dict[str, Callable] = {
            'grad_dot': dot,
            'grad_cos': cos,
        }

        if sim_fn not in sim_fn_opts.keys():
            raise ValueError(f"""Unknown method {sim_fn}. Consider using: '{"' | '".join(sim_fn_opts.keys())}'.""")

        self.sim_fn = sim_fn_opts[sim_fn]
        self.kshap = kshap

    @property
    def predictor(self):
        return self.kshap.predictor

    @predictor.setter
    def predictor(self, value):
        self.kshap.predictor = value

    def fit(self, X_train: np.ndarray) -> 'ShapSimilarity':
        """
        Fit the explainer.

        Parameters
        ----------
        X_train
            Training data of shape `(N, F)`, where `N` is the number of instances and `F` is the number of features.
        """

        # compute shap explanations
        exp_train = self.kshap.explain(X=X_train)

        self.X_train = X_train
        self.y_train = exp_train.data['raw']['prediction']
        self.sv_train = exp_train.data['shap_values']
        return self

    def explain(self, X: np.ndarray):
        """
        Explains the predictor's prediction for a given input.

        Parameters
        ---------
        X
            A array of shape `(M, F)`, where `M` is the number of instances to be explained and `F` is the number
            of feature.

        Returns
        -------
        `Explanation` object.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        exp_test = self.kshap.explain(X=X)
        y = exp_test.data['raw']['prediction']
        sv_test = exp_test.data['shap_values']

        # define buffers
        scores = []
        ordered_indices = []
        most_similar, least_similar = [], []

        # compute similarity for each instance. This has to be done one at the time since we are computing
        # the shapley values w.r.t. the predicted output and not w.r.t. to a given loss function.
        for i, pred in enumerate(y):
            sv_train_tmp = self.sv_train[pred]
            sv_test_tmp = sv_test[pred][i:i+1]

            # compute scores
            scores_tmp = self.sim_fn(sv_train_tmp, sv_test_tmp).reshape(-1)

            # order scores in descending order
            ordered_indices_tmp = np.argsort(scores_tmp)[::-1]
            scores_tmp = scores_tmp[ordered_indices_tmp]

            most_similar_instances_tmp = self.X_train[ordered_indices_tmp[:ShapSimilarity.NUM_RETURNED]]
            most_similar_labels_tmp = self.y_train[ordered_indices_tmp[:ShapSimilarity.NUM_RETURNED]]
            most_similar_scores_tmp = scores_tmp[:ShapSimilarity.NUM_RETURNED]

            least_similar_instances_tmp = self.X_train[ordered_indices_tmp[-ShapSimilarity.NUM_RETURNED:]]
            least_similar_labels_tmp = self.y_train[ordered_indices_tmp[-ShapSimilarity.NUM_RETURNED:]]
            least_similar_scores_tmp = scores_tmp[-ShapSimilarity.NUM_RETURNED:]

            # append data to buffers
            scores.append(scores_tmp)
            ordered_indices.append(ordered_indices_tmp)
            most_similar.append({
                "instances": most_similar_instances_tmp,
                "labels": most_similar_labels_tmp,
                "scores": most_similar_scores_tmp
            })
            least_similar.append({
                "instances": least_similar_instances_tmp,
                "labels": least_similar_labels_tmp,
                "scores": least_similar_scores_tmp
            })

        return self._build_explanation(X=X,
                                       y=y,
                                       scores=scores,
                                       ordered_indices=ordered_indices,
                                       most_similar=most_similar,
                                       least_similar=least_similar)

    def _build_explanation(self,
                           X: np.ndarray,
                           y: np.ndarray,
                           scores: List[np.ndarray],
                           ordered_indices: List[np.ndarray],
                           most_similar: Dict[str, np.ndarray],
                           least_similar: Dict[str, np.ndarray]) -> Explanation:

        """
        Builds the explanation object.

        Parameters
        ----------
        X
            Array of instances to be displayed of shape `(M, F)`, where `M` is the number of instances to be explained
            and `F` is the number of features.
        y
            Array of predictions of shape `(M, )`, where `M` is the number of instance to be explained.
        scores
            An array of shape `(M, N)` containing the similarity scores ordered in descending order along axis 1,
            where `M` is the number of instances to be explained and `N` is the number of train instances.
        ordered_indices
            An array of shape `(M, N)` containing the indices of the most similar instance from the train set,
            sorted according to the `scores`, where `M` is the number of instances to be explained and `N` is the
            number of train instances.
        most_similar
            An array of shape `(M, 5, F)` containing most 5 similar instance, where `M` is the number of instances
            to be explained, and `F` is the number of features.
        least_similar
            An array of shape `(M, 5, F)` containing the least 5 similar instance, where `M` is the number of instances
            to be explained, and `F` is the number of features.

        Returns
        -------
        `Explanation` object.
        """
        data = copy.deepcopy(DEFAULT_DATA_SHAP_SIM)
        data.update(X=X, y=y,
                    scores=scores,
                    ordered_indices=ordered_indices,
                    most_similar=most_similar,
                    least_similar=least_similar)
        return Explanation(meta=self.meta, data=data)

    def reset_predictor(self, predictor: Any) -> None:
        self.kshap.reset_predictor(predictor)
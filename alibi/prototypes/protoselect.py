import os
import logging
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from tqdm import tqdm
from copy import deepcopy
from typing import Callable, Optional, Dict, List, Union, Any, Tuple
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize

from alibi.utils.distance import batch_compute_kernel_matrix
from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.api.defaults import DEFAULT_META_PROTOSELECT, DEFAULT_DATA_PROTOSELECT
from alibi.utils.kernel import EuclideanDistance

logger = logging.getLogger(__name__)


class ProtoSelect(Explainer, FitMixin):
    def __init__(self,
                 eps: float,
                 kernel_distance: Callable,
                 lbd: float = None,
                 batch_size: int = int(1e10),
                 preprocess_fn: Callable = None,
                 verbose: bool = False,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------
        kernel_distance
            Kernel to be used. Use `GaussianRBFDistance` or `EuclideanDistance`.
        eps
            Epsilon ball size.
        lbd
            Penalty for each prototype. Encourages a lower number of prototypes to be selected.
        batch_size
            Batch size to be used for kernel matrix computation.
        preprocess_fn
            Preprocessing function for kernel matrix computation.
        verbose
             Whether to display progression bar while computing prototypes points.
        """
        super().__init__(meta=deepcopy(DEFAULT_META_PROTOSELECT))
        self.kernel_distance = kernel_distance
        self.eps = eps
        self.lbd = lbd
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.verbose = verbose

        # get kernel tag
        if hasattr(self.kernel_distance, '__name__'):
            kernel_distance_tag = self.kernel_distance.__name__
        elif hasattr(self.kernel_distance, '__class__'):
            kernel_distance_tag = self.kernel_distance.__class__
        else:
            kernel_distance_tag = 'unknown kernel distance'

        # update metadata
        self.meta['params'].update({
            'kernel_distance': kernel_distance_tag,
            'eps': eps,
            'lbd': lbd,
            'batch_size': batch_size,
            'verbose': verbose
        })

    def fit(self,
            X: np.ndarray,
            X_labels: Optional[np.ndarray] = None,
            Y: Optional[np.ndarray] = None) -> 'ProtoSelect':
        """
        Fit the explainer by setting the reference dataset.

        Parameters
        ---------
        X
            Reference dataset to be summarized.
        X_labels
            Labels of the reference dataset.
        Y
            Dataset to choose the prototypes from. If ``None``, the prototypes will be selected from the reference
            dataset `X`.

        Returns
        -------
        self
            Reference to itself.
        """
        self.X = X
        # if the `X_labels` are not provided, then consider that all elements belong to the same class. This means
        # that loss term which tries to avoid including in an epsilon ball elements belonging to other classes
        # will always be 0. Still the first term of the loss tries to cover as many examples as possible with
        # minimal overlap between the epsilon balls corresponding to the other prototypes.
        self.X_labels = X_labels.astype(np.int32) if (X_labels is not None) else np.zeros((len(X), ), dtype=np.int32)
        # if the set of prototypes is not provided, then find the prototypes belonging to the reference dataset.
        self.Y = Y if (Y is not None) else self.X
        # initialize penalty for adding a prototype
        if self.lbd is None:
            self.lbd = 1 / len(self.X)
            self.meta['params'].update({'lbd': self.lbd})

        self.max_label = np.max(self.X_labels)
        self.kmatrix_yx = batch_compute_kernel_matrix(x=self.Y,
                                                      y=self.X,
                                                      kernel=self.kernel_distance,
                                                      batch_size=self.batch_size,
                                                      preprocess_fn=self.preprocess_fn)

        return self

    def explain(self, num_prototypes: int = 1) -> Explanation:
        """
        Searches for the requested number of prototypes. Note that the algorithm can return a lower number of
        prototypes than the requested one. To increase the number of prototypes, reduce the epsilon-ball radius
        `eps` and the penalty `lbd` for adding a prototype.

        Parameters
        ----------
        num_prototypes
            Number of prototypes to be selected.

        Returns
        -------
        An `Explanation` object containing the prototypes, prototypes indices and protoypes labels with additional \
        metadata as attributes
        """
        if num_prototypes > len(self.Y):
            num_prototypes = len(self.Y)
            logger.warning('The number of prototypes requested is larger than the number of elements from '
                           f'the prototypes selection set. Automatically setting `num_prototypes={num_prototypes}`.')

        # dictionary of prototypes indices for each class
        protos = {l: [] for l in range(self.max_label + 1)}
        # set of available prototypes indices. Note that initially we start with the entire set of Y,
        # but as the algorithm progresses, we remove the indices of the prototypes that we already selected.
        available_indices = set(range(len(self.Y)))
        # matrix of size `[NY, NX]`, where `NY = len(Y)` and `NX = len(X)`
        # represents a mask which indicates for each element `y` in `Y` what are the elements of `X` that are in an
        # epsilon ball centered in `y`.
        B = (self.kmatrix_yx <= self.eps).astype(np.int32)
        # matrix of size `[L, NX]`, where `L` is the number of labels
        # each row `l` indicates the elements from `X` that are covered by prototypes belonging to class `l`
        B_P = np.zeros((self.max_label + 1, len(self.X)), dtype=np.int32)
        # matrix of size `[L, NX]`. Each row `l` indicates which elements form `X` are labeled as `l`
        Xl = np.concatenate([(self.X_labels == l).reshape(1, -1)
                             for l in range(self.max_label + 1)], axis=0).astype(np.int32)

        # vectorized implementation of the prototypes scores.
        # See paper (pag 8): https://arxiv.org/pdf/1202.5933.pdf for more details
        B_diff = B[:, np.newaxis, :] - B_P[np.newaxis, :, :]  # [NY, 1, NX] - [1, L, NX] -> [NY, L, NX]
        # [NY, L, NX] + [1, L, NX] -> [NY, L, NX]
        delta_xi_all = B_diff + Xl[np.newaxis, ...] >= 2
        # [NY, L]. For every row `y` and every column `l`, we compute how many new instances belonging to class
        # `l` will be covered if we add the prototype `y`.
        delta_xi_summed = np.sum(delta_xi_all, axis=-1)
        # [NY, 1, NX] +  [1, L, NX] -> [NY, L, NX]
        delta_nu_all = B[:, np.newaxis, :] + (1 - Xl[np.newaxis, ...]) >= 2
        # [NY, L]. For every row `y` and every column `l`, we compute how many new instances belonging to all the
        # other classes different then `l` will be covered if we add the prototype `y`.
        delta_nu_summed = np.sum(delta_nu_all, axis=-1)
        # compute the tradeoff score - each prototype tries to cover as many new elements as possible
        # belonging to the same class, while trying to avoid covering elements belonging to another class
        scores_all = delta_xi_summed - delta_nu_summed - self.lbd

        # add progressing bar if `verbose=True`
        generator = range(num_prototypes)
        if self.verbose:
            generator = tqdm(generator)

        for _ in generator:
            j = np.array(list(available_indices))
            scores = scores_all[j]

            # stopping criterion. The number of the returned prototypes might be lower than
            # the number of requested prototypes
            if np.all(scores < 0):
                break

            # find the index `i` of the best prototype and the class `l` that it covers
            row, col = np.unravel_index(np.argmax(scores), scores.shape)
            i, l = j[row], col

            # update the score
            covered = np.sum(delta_xi_all[:, l, B[i].astype(bool)], axis=-1)
            delta_xi_all[:, l, B[i].astype(bool)] = 0
            delta_xi_summed[:, l] -= covered
            scores_all[:, l] -= covered

            # add prototype to the corresponding list according to the class label `l` that it covers
            # and remove the index `i` from list of available indices
            protos[l].append(i)
            available_indices.remove(i)

        return self._build_explanation(protos)

    def _build_explanation(self, protos: Dict[int, List[int]]) -> Explanation:
        """
        Helper method to build `Explanation` object.
        """
        data = deepcopy(DEFAULT_DATA_PROTOSELECT)
        data['prototypes_indices'] = np.concatenate(list(protos.values())).astype(np.int32)
        data['prototypes_labels'] = np.concatenate([[l] * len(protos[l]) for l in protos]).astype(np.int32)
        data['prototypes'] = self.Y[data['prototypes_indices']]
        return Explanation(meta=self.meta, data=data)

    def save(self, path: Union[str, os.PathLike]) -> None:
        super().save(path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike], predictor: Optional[Any] = None) -> "Explainer":
        return super().load(path, predictor)


def _helper_protoselect_euclidean_1knn(explainer: ProtoSelect,
                                       num_prototypes: int,
                                       eps: float) -> KNeighborsClassifier:
    """
    Helper function to fit a 1-KNN classifier on the prototypes returned by the explainer.
    Sets the epsilon radius to be used.

    Parameters
    ----------
    explainer
        Fitted explainer.
    num_prototypes
        Number of requested prototypes.
    eps
        Epsilon radius to be set and used for the computation of prototypes.

    Returns
    -------
    Fitted KNN-classifier.
    """
    # update explainer eps and get explanation
    explainer.eps = eps
    explanation = explainer.explain(num_prototypes=num_prototypes)

    # train 1-knn classifier
    proto, proto_labels = explanation.data['prototypes'], explanation.data['prototypes_labels']
    knn = KNeighborsClassifier(n_neighbors=1)
    return knn.fit(X=proto, y=proto_labels)


def cv_protoselect_euclidean(refset: Tuple[np.ndarray, np.ndarray],
                             protoset: Tuple[np.ndarray, ],
                             valset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                             num_prototypes: int = 1,
                             eps_range: Optional[np.ndarray] = None,
                             quantiles: Optional[Tuple[float, float]] = None,
                             grid_size: int = 25,
                             n_splits: int = 2,
                             batch_size: int = int(1e10),
                             preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                             **kwargs) -> float:
    """
    Cross-validation parameter selection for ProtoSelect with Euclidean distance. The method computes
    the best epsilon radius.

    Parameters
    ----------
    refset
        Tuple, `(X_ref, X_ref_labels)`, consisting of the reference data instances with the corresponding reference
        labels.
    protoset
        Tuple, `(X_proto, )`, consisting of the prototypes selection set. Note that the argument is passed as a tuple
        with a single element for consistency reasons.
    valset
        Optional tuple `(X_val, X_val_labels)` consisting of validation data instances with the corresponding
        validation labels. 1-KNN classifier is evaluated on the validation dataset to obtain the best epsilon radius.
        In case ``valset=None``, then `n-splits` cross-validation is performed on the `refset`.
    num_prototypes
        The number of prototypes to be selected.
    eps_range
        Optional ranges of values to select the epsilon radius from. If not specified, the search range is
        automatically proposed based on the inter-distances between `X_ref` and `X_proto`. The distances are filtered
        by considering only values in between the `quantiles` values. The minimum and maximum distance values are
        used to define the range of values to search the epsilon radius. The interval is discretized in `grid_size`
        equal-distant bins.
    quantiles
        Quantiles, `(q_min, q_max)`, to be used to filter the range of values of the epsilon radius. See `eps_range` for
        more details. If not specified, no filtering is applied. Only used if ``eps_range=None``.
    grid_size
        The number of equal-distant bins to be used to discretize the `eps_range` proposed interval. Only used if
        ``eps_range=None``.
    n_splits
        The number of cross-validation splits to be used. Default value 2. Only used if ``valset=None``.
    batch_size
        Batch size to be used for kernel matrix computation.
    preprocess_fn
        Preprocessing function to be applied to the data instance before applying the kernel.

    Returns
    -------
    Best epsilon radius according to the accuracy of a 1-KNN classifier.
    """
    # unpack datasets
    X_ref, X_ref_labels = refset
    X_proto = protoset[0]

    if preprocess_fn is not None:
        X_ref = _batch_preprocessing(X=X_ref, preprocess_fn=preprocess_fn, batch_size=batch_size)
        X_proto = _batch_preprocessing(X=X_proto, preprocess_fn=preprocess_fn, batch_size=batch_size)

    # propose eps_range if not specified
    if eps_range is None:
        dist = batch_compute_kernel_matrix(x=X_ref, y=X_proto, kernel=EuclideanDistance()).reshape(-1)
        if quantiles is not None:
            if quantiles[0] > quantiles[1]:
                raise ValueError('The quantile lower-bound is greater then the quantile upper-bound.')

            quantiles = np.clip(quantiles, a_min=0, a_max=1)
            min_dist, max_dist = np.quantile(a=dist, q=quantiles)
        else:
            min_dist, max_dist = np.min(dist), np.max(dist)
        # define list of values for eps
        eps_range = np.linspace(min_dist, max_dist, num=grid_size)

    if valset is None:
        kf = KFold(n_splits=n_splits)
        scores = np.zeros((grid_size, n_splits))

        for i, (train_index, val_index) in enumerate(kf.split(X=X_ref, y=X_ref_labels)):
            X, X_labels = X_ref[train_index], X_ref_labels[train_index]
            X_val, X_val_labels = X_ref[val_index], X_ref_labels[val_index]

            # define and fit explainer here, so we don't repeat the kernel matrix computation in the next for loop
            explainer = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0, **kwargs)
            explainer = explainer.fit(X=X, X_labels=X_labels, Y=X_proto)

            for j in range(grid_size):
                knn = _helper_protoselect_euclidean_1knn(explainer=explainer,
                                                         num_prototypes=num_prototypes,
                                                         eps=eps_range[j])
                X_val_ft = preprocess_fn(X_val) if (preprocess_fn is not None) else X_val
                scores[j][i] = knn.score(X_val_ft, X_val_labels)

        # compute mean score across splits
        scores = np.mean(scores, axis=-1)
    else:
        scores = np.zeros(grid_size)
        X_val, X_val_labels = valset

        if preprocess_fn is not None:
            X_val = _batch_preprocessing(X=X_val, preprocess_fn=preprocess_fn, batch_size=batch_size)

        # define and fit explainer, so we don't repeat the kernel matrix computation
        explainer = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0, **kwargs)
        explainer = explainer.fit(X=X_ref, X_labels=X_ref_labels, Y=X_proto)

        for j in range(grid_size):
            knn = _helper_protoselect_euclidean_1knn(explainer=explainer,
                                                     num_prototypes=num_prototypes,
                                                     eps=eps_range[j])

            X_val_ft = preprocess_fn(X_val) if (preprocess_fn is not None) else X_val
            scores[j] = knn.score(X_val_ft, X_val_labels)

    return eps_range[np.argmax(scores)]


def _batch_preprocessing(X: np.ndarray,
                         preprocess_fn: Callable[[np.ndarray], np.ndarray],
                         batch_size: int = 32) -> np.ndarray:
    """
    Preprocess a dataset `X` in batches by applying the preprocessor function.

    Parameters
    ----------
    X
        Dataset to be preprocessed.
    preprocess_fn
        Preprocessor function.
    batch_size
        Batch size to be used for each call to `preprocess_fn`.

    Returns
    -------
    Preprocessed dataset.
    """
    X_ft = []
    num_iter = int(np.ceil(len(X) / batch_size))

    for i in range(num_iter):
        istart, iend = batch_size * i, min(batch_size * (i + 1), len(X))
        X_ft.append(preprocess_fn(X[istart:iend]))

    return np.concatenate(X_ft, axis=0)


def _imscatterplot(x: np.ndarray,
                   y: np.ndarray,
                   images: np.ndarray,
                   figsize: Tuple[int, int],
                   image_size: Tuple[int, int] = (28, 28),
                   zoom: Optional[np.ndarray] = None,
                   zoom_lb: float = 1.0,
                   zoom_ub=2.0,
                   sort_by_zoom: bool = True) -> None:
    """
    2D image scatter plot.

    Parameters
    ----------
    x
        Images' x-coordinates.
    y
        Images' y-coordinates.
    images
        Array of images to be placed at coordinates `(x, y)`.
    figsize
        `Matplotlib` figure size.
    image_size
        Size of the generated output image as `(rows, cols)`.
    zoom
        Images' zoom to be used.
    zoom_lb
        Zoom lower bound. The zoom values will be scaled linearly between `[zoom_lb, zoom_up]`.
    zoom_ub
        Zoom upper bound. The zoom values will be scaled linearly between `[zoom_lb, zoom_up]`.
    """
    if zoom is None:
        zoom = np.ones(len(images))

    zoom_min, zoom_max = np.min(zoom), np.max(zoom)
    zoom = (zoom - zoom_min) / (zoom_max - zoom_min) * (zoom_ub - zoom_lb) + zoom_lb

    if sort_by_zoom:
        idx = np.argsort(zoom)[::-1]
        zoom = zoom[idx]
        x, y, images = x[idx], y[idx], images[idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])

    images = [resize(images[i], image_size) for i in range(len(images))]
    imgs = [OffsetImage(img, zoom=zoom[i], cmap='gray') for i, img in enumerate(images)]
    artists = []

    for i in range(len(imgs)):
        x0, y0, im = x[i], y[i], imgs[i]
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def visualize_prototypes(explanation: 'Explanation',
                         refset: Tuple[np.ndarray, np.ndarray],
                         reducer: Callable[[np.ndarray], np.ndarray],
                         preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                         figsize: Tuple[int, int] = (10, 10),
                         image_size: Tuple[int, int] = (28, 28),
                         zoom_lb: float = 1.0,
                         zoom_ub: float = 3.0):
    """
    Plot the images of the prototypes at the location given by the `reducer` representation.
    The size of each prototype is proportional to the log of the number of correct-class training images covered
    by that prototype (Bien and Tibshiran (2012): https://arxiv.org/abs/1202.5933).

    Parameters
    ----------
    explanation
        Explanation object.
    refset
        Tuple, `(X_ref, X_ref_labels)`, consisting of the reference data instances with the corresponding reference
        labels.
    reducer
        2D reducer. Reduces the input feature representation to 2D. Note that the reducer operated directly on the
        input instances if ``preprocess_fn=None``. If the `preprocess_fn` is specified, the reducer will be called
        on the feature representation obtained after calling `preprocess_fn` on the input instances.
    figsize
        `Matplotlib` figure size.
    image_size
        Shape to which the prototype images will be resized. A zoom of 1 will display the image having the shape
        `image_size`.
    zoom_lb
        Zoom lower bound. The zoom will be scaled linearly between `[zoom_lb, zoom_ub]`.
    zoom_ub
        Zoom upper bound. The zoom will be scaled linearly between `[zoom_lb, zoom_ub]`.
    """
    X_ref, X_ref_labels = refset
    X_proto = explanation.data['prototypes']
    X_proto_labels = explanation.data['prototypes_labels']

    # preprocess the dataset
    X_ref_ft = _batch_preprocessing(X=X_ref, preprocess_fn=preprocess_fn) \
        if (preprocess_fn is not None) else X_ref
    X_proto_ft = _batch_preprocessing(X=X_proto, preprocess_fn=preprocess_fn) \
        if (preprocess_fn is not None) else X_proto

    # train knn classifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn = knn.fit(X=X_proto_ft, y=X_proto_labels)

    # get neighbors indices for each training instance
    neigh_idx = knn.kneighbors(X=X_ref_ft, n_neighbors=1)[1].reshape(-1)

    # compute how many training instances each prototype covers
    idx, counts = np.unique(neigh_idx, return_counts=True)
    covered = {i: c for i, c in zip(idx, counts)}

    # compute how many correct labeled instances each prototype covers
    idx, counts = np.unique(neigh_idx[X_proto_labels[neigh_idx] == X_ref_labels], return_counts=True)
    correct = {i: c for i, c in zip(idx, counts)}

    # compute zoom
    zoom = np.log([correct.get(i, 0) for i in covered])

    # compute 2D embedding
    X_protos_2d = reducer(X_proto_ft)
    x, y = X_protos_2d[:, 0], X_protos_2d[:, 1]

    # plot images
    _imscatterplot(x=x, y=y, images=X_proto, figsize=figsize, image_size=image_size,
                   zoom=zoom, zoom_lb=zoom_lb, zoom_ub=zoom_ub)


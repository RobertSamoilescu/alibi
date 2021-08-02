from alibi.explainers.backends.cfrl_tabular import split_ohe, generate_condition  # noqa: F401

# The following methods are included since `alibi.explainers.backends.pytorch.cfrl_tabular` is an extension to the
# `alibi.explainers.backends.pytorch.cfrl_base.py`. In the explainer class `alibi.explainers.cfrl_tabular` the
# access to the backend specific methods is performed through `self.backend` which is of `types.ModuleType`. Since
# some of the methods imported below are common for both data modalities and are access through `self.backend`
# we import them here, without being used explicitly in this module.

from alibi.explainers.backends.tensorflow.cfrl_base import get_actor, get_critic, get_optimizer, data_generator, \
    encode, decode, generate_cf, update_actor_critic, add_noise, to_numpy, to_tensor, set_seed, \
    save_model, load_model  # noqa: F403, F401

import torch
import torch.nn.functional as F
from typing import List, Dict


def sample_differentiable(X_ohe_hat_split: List[torch.Tensor],
                          category_map: Dict[int, List[str]]) -> List[torch.Tensor]:
    """
    Samples differentiable reconstruction.

    Parameters
    ----------
    X_ohe_hat_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.

    Returns
    -------
    Differentiable reconstruction.
    """
    num_attr = len(X_ohe_hat_split) - len(category_map)
    cat_attr = len(category_map)
    X_out = []

    # pass numerical attributes as they are
    if num_attr > 0:
        X_out.append(X_ohe_hat_split[0])

    # sample categorical attributes
    if cat_attr > 0:
        for head in X_ohe_hat_split[-cat_attr:]:
            out = torch.argmax(head, dim=1)

            # transform to one-hot encoding
            out = F.one_hot(out, num_classes=head.shape[1])
            proba = F.softmax(head, dim=1)
            out = out - proba.detach() + proba
            X_out.append(out)

    return X_out


def l0_ohe(input: torch.Tensor,
           target: torch.Tensor,
           reduction: str = 'none') -> torch.Tensor:
    """
    Computes the L0 loss for a one-hot encoding representation.

    Parameters
    ----------
    input
        Input tensor.
    target
        Target tensor
    reduction
        Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

    Returns
    -------
    L0 loss.
    """
    loss = torch.maximum(target - input, torch.zeros_like(input))

    if reduction == 'none':
        return loss

    if reduction == 'mean':
        return torch.mean(loss)

    if reduction == 'sum':
        return torch.sum(loss)

    raise ValueError(f"Reduction {reduction} not implemented.")


def l1_loss(input: torch.Tensor, target: torch.Tensor, reduction: str = 'none') -> torch.Tensor:
    """
    Computes L1 loss.

    Parameters
    ----------
    input
        Input tensor.
    target
        Target tensor.
    reduction
        Specifies the reduction to apply to the output: `none` | `mean` | `sum`.

    Returns
    -------
    L1 loss.
    """
    return F.l1_loss(input=input, target=target, reduction=reduction)


def sparsity_loss(X_ohe_hat_split: List[torch.Tensor],
                  X_ohe: torch.Tensor,
                  category_map: Dict[int, List[str]],
                  weight_num: float = 1.0,
                  weight_cat: float = 1.0):
    """
    Computes heterogeneous sparsity loss.

    Parameters
    ----------
    X_ohe_hat_split
        List of one-hot encoded reconstructed columns form the auto-encoder.
    X_ohe
        One-hot encoded representation of the input.
    category_map
        Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
        values for an attribute.
    weight_num
        Numerical loss weight.
    weight_cat
        Categorical loss weight.

    Returns
    -------
    Heterogeneous sparsity loss.
    """
    # split the input into a list of tensor, where each element corresponds to a network head
    X_ohe_num_split, X_ohe_cat_split = split_ohe(X_ohe=X_ohe,
                                                 category_map=category_map)

    # sample differentiable output
    X_ohe_hat_split = sample_differentiable(X_ohe_hat_split=X_ohe_hat_split,
                                            category_map=category_map)

    # define numerical and categorical loss
    num_loss, cat_loss = torch.tensor(0.), torch.tensor(0.)
    offset = 0

    # compute numerical loss
    if len(X_ohe_num_split) > 0:
        offset = 1
        num_loss = torch.mean(l1_loss(input=X_ohe_hat_split[0],
                                      target=X_ohe_num_split[0],
                                      reduction='none'))

    # compute categorical loss
    if len(X_ohe_cat_split) > 0:
        for i in range(len(X_ohe_cat_split)):
            batch_size = X_ohe_hat_split[i].shape[0]
            cat_loss += torch.sum(l0_ohe(input=X_ohe_hat_split[i + offset],
                                         target=X_ohe_cat_split[i],
                                         reduction='none')) / batch_size

        cat_loss /= len(X_ohe_cat_split)

    return {"num_loss": weight_num * num_loss, "cat_loss": weight_cat * cat_loss}


def consistency_loss(Z_cf_pred: torch.Tensor, Z_cf_tgt: torch.Tensor, **kwargs):
    """
    Computes heterogeneous consistency loss.

    Parameters
    ----------
    Z_cf_pred
        Predicted counterfactual embedding.
    x_cf
        Counterfactual reconstruction. This should be already post-processed.

    Returns
    -------
    Heterogeneous consistency loss.
    """
    # compute consistency loss
    loss = F.mse_loss(Z_cf_pred, Z_cf_tgt)
    return {"consistency_loss": loss}

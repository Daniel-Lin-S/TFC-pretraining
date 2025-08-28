import numpy as np
import torch

from .utils import one_hot_encoding


def DataTransform(sample, config):
    """
    Strong augmentations: Jittering + Permutation
    Weak augmentations: Scaling

    Parameters
    ----------
    sample : numpy.ndarray  
        The input time series data.
    config : object
        Configuration object containing augmentation parameters.
        Required attributes:
        - augmentation.jitter_ratio: float, standard deviation for jittering.
        - augmentation.jitter_scale_ratio: float, standard deviation for scaling.
        - augmentation.max_seg: int, maximum number of segments for permutation.
    """
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)

    strong_aug = jitter(
        permutation(sample, max_segments=config.augmentation.max_seg).numpy(),
        config.augmentation.jitter_ratio
    )

    return weak_aug, strong_aug

def DataTransform_TD(sample, config):
    """
    Simplely use the jittering augmentation.
    In TF-C framework, the augmentation has litter impact on the final tranfering performance.

    Parameters
    ----------
    sample : numpy.ndarray
        The input time series data.
    config : object
        Configuration object containing augmentation parameters.
    """
    aug = jitter(sample, config.augmentation.jitter_ratio)

    return aug


def DataTransform_TD_bank(sample, config):
    """
    Augmentation bank that includes all 4 augmentations and randomly select one as the positive sample.
    You may use this one the replace the above DataTransform_TD function.

    Parameters
    ----------
    sample : numpy.ndarray
        The input time series data.
    config : object
        Configuration object containing augmentation parameters.
        Required attributes:
        - augmentation.jitter_ratio: float, standard deviation for jittering.
        - augmentation.jitter_scale_ratio: float, standard deviation for scaling.
        - augmentation.max_seg: int, maximum number of segments for permutation.
    """
    aug_1 = jitter(sample, config.augmentation.jitter_ratio)
    aug_2 = scaling(sample, config.augmentation.jitter_scale_ratio)
    aug_3 = permutation(sample, max_segments=config.augmentation.max_seg)
    aug_4 = masking(sample, keepratio=0.9)

    li = np.random.randint(0, 4, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    aug_1 = aug_1 * li_onehot[:, 0][:, None, None]  # the rows that are not selected are set as zero.
    aug_2 = aug_2 * li_onehot[:, 0][:, None, None]
    aug_3 = aug_3 * li_onehot[:, 0][:, None, None]
    aug_4 = aug_4 * li_onehot[:, 0][:, None, None]
    aug_T = aug_1 + aug_2 + aug_3 + aug_4
    return aug_T

def DataTransform_FD(sample, config):
    """
    Weak and strong augmentations in Frequency domain

    Parameters
    ----------
    sample : torch.Tensor
        The input frequency domain data.
    config : object
        Configuration object containing augmentation parameters.
        Currently a placeholder.

    Returns
    -------
    torch.Tensor
        The augmented frequency domain data.
    """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F

def remove_frequency(x: torch.Tensor, pertub_ratio: float=0.0):
    """
    Randomly remove part of the frequency components.

    Parameters
    ----------
    x : torch.Tensor
        The input frequency domain data.
        Shape: (num_samples, num_frequency_bins).
    pertub_ratio : float, optional
        Ratio of frequency components to be removed.
        Default: 0.0 (no components removed).

    Returns
    -------
    torch.Tensor
        The frequency domain data with some components removed.

    References
    ----------
    Zhang, Xiang, et al.
    "Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency."
    Proceedings of Neural Information Processing Systems (NeurIPS), 2022.
    """
    mask = torch.empty(x.shape, device=x.device).uniform_() > pertub_ratio
    mask = mask.to(x.device)
    return x*mask

def add_frequency(
        x: torch.Tensor, pertub_ratio: float=0.0,
        noise_level: float=0.1
    ) -> torch.Tensor:
    """
    Randomly perturb the frequency components by adding noise.

    Parameters
    ----------
    x : torch.Tensor
        The input frequency domain data.
    pertub_ratio : float, optional
        Ratio of frequency components to be perturbed.
        Default: 0.0 (no components perturbed).
    noise_level : float, optional
        Maximum amplitude of the noise as a fraction of the maximum amplitude of the input signal.
        Default: 0.1 (10% of the maximum amplitude).
    """

    mask = torch.empty(x.shape, device=x.device).uniform_() > (1-pertub_ratio)
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape) * (max_amplitude * noise_level)
    pertub_matrix = mask * random_am
    return x + pertub_matrix


def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)

def masking(x: torch.Tensor, keepratio: float=0.9) -> torch.Tensor:
    """
    Randomly mask parts of the input time series.
    Binary mask is generated from a binomial distribution.

    Parameters
    ----------
    x : torch.Tensor
        The input time series data.
        Shape: (batch_size, seq_length, num_channels).
    keepratio : float
        Ratio of the time series to keep (not mask).

    Returns
    -------
    torch.Tensor
        The masked time series data, with the same shape as input.
    """
    global mask_id
    nan_mask = ~x.isnan().any(axis=-1) # type: ignore
    x[~nan_mask] = 0

    mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=keepratio).to(x.device)
    # mask &= nan_mask
    x[~mask_id] = 0
    return x

def jitter(x: np.ndarray, sigma: float=0.8) -> np.ndarray:
    """
    Randomly perturb the signal by adding Gaussian noise.

    Parameters
    ----------
    x : numpy.ndarray
        The input time series data.
        Shape: not restricted.
    sigma : float
        Standard deviation of the Gaussian noise to be added.

    Returns
    -------
    numpy.ndarray
        The perturbed time series data of the same shape as input.

    References
    ----------
    Um, Terry T., et al.
    “Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks.”
    Proceedings of the 19th ACM International Conference on Multimodal Interaction, ACM, 2017, pp. 216-20.
    Crossref, https://doi.org/10.1145/3136755.3136817.
    """
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x: np.ndarray, sigma: float=1.1) -> np.ndarray:
    """
    Randomly scale the signal.

    Parameters
    ----------
    x : numpy.ndarray
        The input time series data.
        Shape: (num_samples, num_channels, seq_length)
    sigma : float
        Standard deviation of the scaling factor.
        Scaling factor is drawn from N(2, sigma)

    Returns
    -------
    numpy.ndarray
        The scaled time series data of the same shape as input.

    References
    ----------
    Um, Terry T., et al.
    “Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks.”
    Proceedings of the 19th ACM International Conference on Multimodal Interaction, ACM, 2017, pp. 216-20.
    Crossref, https://doi.org/10.1145/3136755.3136817.
    """
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def permutation(
        x: np.ndarray,
        max_segments:int=5,
        seg_mode:str="random"
    ) -> torch.Tensor:
    """
    Randomly permute segments of the time series.

    Parameters
    ----------
    x : np.ndarray
        The input time series data.
        Shape: (num_samples, num_channels, seq_length)
    max_segments : int
        Maximum number of segments to split the time series into.
        The actual number of segments is randomly chosen between 1 and max_segments.
    seg_mode : str
        If "random", segments are of random length.
        If "equal", segments are of equal length.
    """
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            elif seg_mode == "equal":
                splits = np.array_split(orig_steps, num_segs[i])
            else:
                raise ValueError(f"Unknown seg_mode: {seg_mode}")

            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat

    return torch.from_numpy(ret)

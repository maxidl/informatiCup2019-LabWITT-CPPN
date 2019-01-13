from cppn_model import CPPN
import numpy as np
from skimage.measure import compare_ssim
import warnings


def similarity(image1: np.array, image2: np.array, color: bool) -> float:
    """
    Calculates the SSIM value between two images.

    Args:
        image1 (np.array): The first image to compare
        image2 (np.array): The second image to compare
        color (bool): Must be True if the given images are RGB, False if they are grayscale.

    Returns:
        float: The SSIM value between the two images
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compare_ssim(image1, image2, multichannel=color)


def mutate(weights: np.array, mutation_prob_weights: float, max_mutation_step: float) -> np.array:
    """
    Mutates each weight value with a given probability by a value sampled from an uniform distribution with the given
    step boundaries.

    Args:
        weights (np.array): The weights to be mutated
        mutation_prob_weights (float): The probability to mutate a weight value
        max_mutation_step (float): the maximum value to add / substract from a weight

    Returns:
        np.array: The mutated weights
    """
    mask = np.random.rand(weights.shape[0]) < mutation_prob_weights
    summand = np.random.uniform(-1.0 * max_mutation_step, max_mutation_step, weights.shape[0])
    new_weights = np.copy(weights)
    new_weights[mask] = (weights + summand)[mask]
    new_weights = np.clip(new_weights, -1.0, 1.0)
    assert weights.shape == new_weights.shape
    return new_weights


def init_cppn_from_img(image: np.array, color: bool = True, sim_threshold: float = 0.6, max_optim_iter: int = 10000,
                       init_stop_bound: int = 100) -> CPPN:
    """
    Initializes a CPPN using the given image by optimizing the SSIM score between the CPPNs output and the image.
    This is done by trying multiple net depths for the CPPN and pre-optimizing its weights using evolutionary strategies.

    Args:
        image (np.array): the image to be optimized for.
        color (bool): Initilize the CPPN for either RGB if True, or grayscale if False.
        sim_threshold: A similarity threshold to prevent the CPPN producing images too similar to the input image.
        max_optim_iter: Maximum number of weight pre-optimization iterations.
        init_stop_bound: Maximum number of CPPNs to try.

    Returns:
        CPPN: The initilized CPPN, which produces images somewhat similar to the input image
    """
    print('Initializing CPPN ...', end='\r')
    cppn = CPPN(color=color, img_size=image.shape[0])
    gen_image = cppn.render_image()
    sim_score = similarity(image, gen_image, color)

    iteration = 0
    not_improved_since = 0
    curr_best = gen_image, cppn.get_weights(), cppn.net_depth, cppn.z

    # print('finding initial CPPN config...')
    # find an initial CPPN that can produce at least somewhat similar images
    while sim_score < sim_threshold and not_improved_since < init_stop_bound:
        new_gen_image = cppn.render_image()
        new_sim_score = similarity(image, new_gen_image, color)
        if new_sim_score > sim_score:
            sim_score = new_sim_score
            # print(f'iter:{iteration}, sim:{sim_score} depth:{cppn.net_depth}')
            curr_best = new_gen_image, cppn.get_weights(), cppn.net_depth, cppn.z
            not_improved_since = 0
        else:
            cppn.reset()
            not_improved_since += 1
        iteration += 1

    # run pre-optimization on sim score up until threshold or max_iter

    # print('pre-optimizing weights...')
    cppn = CPPN(color=color, net_depth=curr_best[2])
    cppn.set_weights(curr_best[1])
    cppn.z = curr_best[3]
    iteration = 0
    optimization_steps = 0
    while sim_score < sim_threshold and iteration < max_optim_iter:
        weights = cppn.get_weights()
        new_weights = mutate(weights, 0.3, 0.05)
        cppn.set_weights(new_weights)
        new_gen_image = cppn.render_image()
        new_sim_score = similarity(image, new_gen_image, color)
        if new_sim_score > sim_score:
            sim_score = new_sim_score
            optimization_steps += 1
            print(iteration, sim_score)
        else:
            cppn.set_weights(weights)  # if no improvement, back to old state
        iteration += 1
        # print(f'{iteration} {sim_score} {new_sim_score}')

    if optimization_steps < 1 and sim_score < 0.3:
        return init_cppn_from_img(image, color)

    # print(f'initialization finished. cppn net_depth:{cppn.net_depth}')
    print('Initializing CPPN ... Done.')
    return cppn

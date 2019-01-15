import sys
import numpy as np
import torch
from util import get_confidence, send_query, save_image, save_gif_from_images, write_to_log, \
    load_image, query_yes_no, clean_filename, image_to_grayscale
from cppn_model import CPPN
from cppn_init import init_cppn_from_img
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from typing import List


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_img', required=True, help='Input image file representing the target class.')
    parser.add_argument('--output_dir', required=True, help='Output directory to save to')
    parser.add_argument('--color', required=False, action='store_true', help='Generate RGB image instead of grayscale.')
    parser.add_argument('--target_conf', required=False, default=0.95, type=float,
                        help="""The targeted confidence value for the adversarial image on the API. Default: 0.95 .
                         Keep in mind that values >0.95 might require a good amount of API queries.""")
    parser.add_argument('--high_res', required=False, action='store_true',
                        help='Output final adversarial in high resolution.')
    parser.add_argument('--max_queries', required=False, default=1000, type=int,
                        help='Maximum number of API queries before aborting.')
    parser.add_argument('--init', required=False, default=False, action='store_true',
                        help='Initialize the CPPN from the input image. Still in experimental status.')
    parser.add_argument('--random_seed', required=False, default=None, type=int,
                        help='Provide a random seed for reproducible results.')
    parser.add_argument('--no_gif', required=False, action='store_true', default=False,
                        help='Disable optimization process .gif output.')
    args = parser.parse_args()
    return args


_mutation_prob_weights = 0.3  # Defines how many weights are changed when mutating
_mutation_step = 0.05  # Defines the maximum value to change a weight when mutating (boundary for uniform sampling)


def __mutate(weights: np.array, try_same_mutation: bool) -> np.array:
    """
    Mutates the given weights. Saves state of last applied mutation.

    Args:
        weights (np.array): The weight array to be mutated.
        try_same_mutation (bool): if True, apply the same mutation as the last time. If False, generates a new mutation.

    Returns:
        np.array: The mutated weights.
    """
    global _mutation, _mask
    if not try_same_mutation:
        _mask = np.random.rand(weights.shape[0]) < _mutation_prob_weights
        _mutation = np.random.uniform(-1.0 * _mutation_step, _mutation_step, weights.shape[0])
    new_weights = np.copy(weights)
    new_weights[_mask] = (weights + _mutation)[_mask]
    new_weights = np.clip(new_weights, -1.0, 1.0)
    return new_weights


def generate_adversarial(target_class: str, target_conf: float = 0.95, target_image: np.array = None,
                         color: bool = True,
                         max_queries=1000,
                         image_size: int = 64,
                         init: bool = False) -> (np.array, float, int, List[np.array], CPPN):
    """
    Generates an adversarial image for the target class using evolution strategies to optimize a CPPN.

    Args:
        target_class (str): The label of the targeted class. Must match a corresponding label in the API.
        target_conf (float): The target confidence. If this threshold is reached the optimization terminates.
        target_image (np.array): If init is True, this image is used to initialize the CPPN.
        color (bool): If True, generate an RGB image. If False, generate a grayscale image.
        max_queries (int): Maximum number of API queries to make. Optimization terminates if this value is reached.
        image_size (int): The image size of the generated adversarial.
        init (bool): If True, initialize the CPPN using the target_image.

    Returns:
        np.array: the generated adversarial image.
        float: the reached confidence.
        int: the number of API queries performed by the optimizer.
        List[np.array]: a list containing all images of the optimization steps.
    """

    def __init_cppn(curr_cppn: CPPN = None):
        # print('\tInitializing CPPN ...')
        if target_image is not None and init:
            cppn = init_cppn_from_img(target_image, color=color)
        else:
            if curr_cppn is None:
                cppn = CPPN(color=color, img_size=image_size)
            else:
                cppn = curr_cppn
                cppn.reset()
        # print('\t\tDone.')
        # save_image('init.png', cppn.render_image())
        return cppn

    # initialization
    min_start_conf = 0.01
    generated_images = []
    final_conv_images = []
    num_queries = 0

    cppn = __init_cppn()

    curr_image = cppn.render_image()
    generated_images.append(curr_image)
    final_conv_images.append(curr_image)
    curr_conf = get_confidence(curr_image, target_class)
    num_queries += 1
    curr_weights = cppn.get_weights()
    try_same_mutation = False
    not_improving_since = 0

    print('\tOptimizing on API ...')
    print('\t\tAPI queries\tconfidence\t')
    print(f'\t\t  {num_queries}\t\t{np.round(curr_conf, 4)}')
    # optimization
    while curr_conf < target_conf:
        if num_queries >= max_queries:
            break

        if curr_conf < min_start_conf:
            # reinitialize
            cppn = __init_cppn(cppn)
            final_conv_images.clear()
            curr_weights = cppn.get_weights()
            curr_image = cppn.render_image()
            generated_images.append(curr_image)
            final_conv_images.append(curr_image)
            curr_conf = get_confidence(curr_image, target_class)
            num_queries += 1
            not_improving_since = 0
            print(f'\t\t  {num_queries}\t\t{np.round(curr_conf, 4)}')
            continue

        # optim step
        new_weights = __mutate(curr_weights, try_same_mutation)
        try_same_mutation = False
        cppn.set_weights(new_weights)
        new_image = cppn.render_image()
        new_conf = get_confidence(new_image, target_class)
        num_queries += 1

        if new_conf > curr_conf:
            # successful step
            not_improving_since = 0
            curr_weights = new_weights
            curr_image = new_image
            generated_images.append(curr_image)
            final_conv_images.append(curr_image)
            if new_conf - curr_conf > 0.01:
                try_same_mutation = True
            curr_conf = new_conf

        else:
            # unsuccessful step
            not_improving_since += 1
            cppn.set_weights(curr_weights)
            if not_improving_since > 100:
                curr_conf = 0  # forces reset
        print(f'\t\t  {num_queries}\t\t{np.round(curr_conf, 4)}')
    return curr_image, np.round(curr_conf, 4), num_queries, final_conv_images, cppn


def main():
    args = parse_args()
    if args.random_seed is not None:
        # fixed random seeds for reproducibility
        np.random.seed(args.random_seed)
        torch.random.manual_seed(args.random_seed)

    # infer target label from image
    input_img = load_image(args.input_img, size=64)

    labels, confidences = send_query(input_img)
    target_idx = np.argmax(confidences)
    target_class = labels[target_idx]

    # ask user if he wants to continue
    print(f'Inferred label: {target_class}, confidence of {np.round(confidences[target_idx], 3)}')
    if not query_yes_no('Continue ?'):
        print('Please choose an input image which the API classifies as your target class. ')
        sys.exit(0)

    # generate adversarial image
    else:
        if not args.color:
            target_img = image_to_grayscale(input_img)
        else:
            target_img = input_img

        print('Generating adversarial...')
        adversarial, conf, num_queries, conv_images, cppn = generate_adversarial(target_class=target_class,
                                                                                 target_image=target_img,
                                                                                 color=args.color,
                                                                                 target_conf=args.target_conf,
                                                                                 max_queries=args.max_queries,
                                                                                 init=args.init)

        if conf < args.target_conf:
            print(f'Failed to generate an adversarial image after {args.max_queries} queries.')
            # write_to_log('log.tsv', f'{target_class}\t{conf}\t{num_queries}\t{args.color}\t{args.init}')
            sys.exit(0)
        print(f'Found an adversarial image with > {args.target_conf} API confidence after {num_queries} queries.')

        output_dir = Path(args.output_dir)
        print(f'\tSaving results in: {output_dir}/')

        # save adversarial image
        adversarial_fname = str(output_dir / f'adversarial_{clean_filename(target_class)}_{conf}')
        save_image(adversarial_fname + '.png', adversarial)
        if args.high_res:
            cppn.set_img_size(2000)
            adversarial_high_res = cppn.render_image()
            save_image(adversarial_fname + '_HD.png', adversarial_high_res)
        # save convergence gif
        if not args.no_gif:
            conv_gif_fname = str(output_dir / f'convergence_{clean_filename(target_class)}_{conf}.gif')
            save_gif_from_images(conv_gif_fname, conv_images)

        # write_to_log('log.tsv', f'{target_class}\t{conf}\t{num_queries}\t{args.color}\t{args.init}')
        print('Finished.')


if __name__ == '__main__':
    main()

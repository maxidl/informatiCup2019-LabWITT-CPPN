#!/usr/bin/env bash
# runs the adversarial generation for each image in the test_images dir
for i in `seq 0 42`
do
    echo ${i}
    yes | python generate_adversarial.py --input_img test_images/${i}.png --output_dir out --high_res --color --random_seed 2
done

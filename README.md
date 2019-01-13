# InformatiCup2019-CPPN

This repository is part of a solution for the [informatiCup 2019](http://www.informaticup.de) competition
hosted by the [Gesellschaft für Informatik](https://gi.de).
The [task](https://github.com/InformatiCup/InformatiCup2019/blob/master/Irrbilder.pdf) 
of the 14th informatiCup is to generate [adversarial examples](https://blog.openai.com/adversarial-example-research/) 
for a given neural network based classification API.

Our solution in this repository tries to solve the use case:
* The attacker does not have access to the dataset used to train the API classifier 
(and also does not have the time to collect a dataset himself)
* He wants to keep the number of necessary API queries as low as possible

This implementation uses [CPPNs](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network)
(Compositional-Pattern-Producing-Networks) optimized with evolutionary strategies.
You can find a nice blog post explaining CPPNs [here](http://blog.otoro.net/2016/03/25/generating-abstract-patterns-with-tensorflow/).

## Example Results
Adversarials (hover to see label):

![aaaa](./examples/adversarial_Ausschlielich_geradeaus_0.9833.png "Ausschließlich geradeaus")


Visualized optimization process:

![](./examples/convergence_Ausschlielich_geradeaus_0.9833.gif "Ausschließlich geradeaus")


See `examples/` for more.

## Usage

#### Download

Clone this repository:
```bash
git clone https://github.com/MaximilianIdahl/InformatiCup2019-CPPN.git
cd InformatiCup2019-CPPN
```
#### Dependencies

This repository includes a conda environment in `environment.yml` as well as a pip environment in `requirements.txt`.
To create a conda environment from this file, simply run
```bash
conda env create -f environment.yml
```
or install the requirements via pip using
```bash
pip install -r requirements.txt
```

#### Prerequisites
Put your API key in the `API_config.ini file`.
```ini
[API]
URL = https://phinau.de/trasi
KEY = <your-api-key>
```
#### Generating Adversarial Images
* You can generate single adversarial images by running:
    ```bash
    python generate_adversarial.py --input_img <some-image-file> --output_dir <some-directory>
    ```
   where <some-image-file> is a image file representing the target class. 
   You can find some example images in the `test_images/` directory.
* To generate an RGB image, include the `--color` flag:
    ```bash
    python generate_adversarial.py --input_img <some-image-file> --output_dir <some-directory> --color
    ```
* Full list of argument options:

    |Option|Required|Default|Description|
    |------|:------:|:-----:|-----------|
    |--input_img|True|-|Input image file representing the target class, e.g. a `.png` file of a stop sign.|
    |--output_dir|True|-|Output directory to save the results.|
    |--color|False|False|Use RGB instead of grayscale.|
    |--target_conf|False|0.95|The targeted confidence value. The optimizer stops after finding an adversarial with a confidence greater than this value.|
    |--high_res|False|False|Output the adversarial image in high resolution (2000 x 2000) as well as in standard resolution.|
    |--max_queries|False|1000|The maximum number of API queries.|
    |--init|False|False|Use the input image to find a good initial CPPN config (e.g. net depth). Still in experimental status.|

* You can run the `run.sh` script to run the program once for each class in the training 
set of the [GTSRB dataset](http://benchmark.ini.rub.de/), i.e. the images contained in `test_images/`. Even though being clearly identifiable by humans, some of those 
images are misclassified by the API, thus leading to wrong target labels when using this script.

* The unittests can be run with
    ```bash
    python test.py
    ```





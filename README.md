## Setup
### Clone the repo
```
git clone https://git.informatik.uni-hamburg.de/lee/qsr-learning.git
cd qsr-learning
git checkout baselines
```

### Setup the environment using `conda` (recommended)
```
conda env create -f environment.yml
conda activate qsr
```

### install the `qsr-learning` package in development mode
`pip install -e .`

### init submodules
`git submodule update --init`

### install the `vr` package in development mode
```
pip install -e systematic-generalization-sqoop
```

## Running Experiments
### Training
`./baselines/train_model_gpu0.sh`
### Testing
`./baselines/test_model_gpu0.sh`

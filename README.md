# monomer-orientation
Model to determine the orientation of a monomer from low-resolution images.

## Setting Up Virtual Env
Create the environment:
`conda env create --file=environment.yml`

Update the Environment:
`conda env update --name monomer --file environment.yml --prune`

## Data Format


## Train Model
`python src/train.py --rebuild --save --plot`
- `--rebuild` flag forces dataset to be rebuilt. Use if first time running or if changes have been made to `src/data.py`
- `--save` flag saves serialized model and configurations objects to `src/models/` directory. The name of the model and configurations follows the format `model-TIMESTAMP.p` and `model-TIMESTAMP_config.p`.
- `--plot` flag outputs a plot for the model loss during training and validation.
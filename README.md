## Conda Environment

To create the `event-nn` environment, run:
```
conda env create -f conda/environment.yml
```

To enable GPU support, instead run:
```
conda env create -f conda/environment_gpu.yml
```

## Code Style

Format all code using the [Black formatter](https://black.readthedocs.io/en/stable/). Use a line limit of 99 characters. To format a file, use the command:
```
black <FILE> --line-length 99
```

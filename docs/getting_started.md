## Virtual Environment
This repository uses the [uv](https://docs.astral.sh/uv/) package manager. To set up the virtual environment with the required dependencies, run:
```bash
uv sync
```

## Environment Variable Configuration
Create a `.env` file in the project root with the required data paths:
```
data_path = 'data'
model_path = 'trained_models'
asset_path = 'assets'
```

Of course, you're free to use your own custom paths, if you wish.

## Pre-commit Hooks
This repository uses [prek](https://github.com/j178/prek) as the framework to run pre-commit actions. Install it with:   
```bash
uv run prek install
```
Hooks enforce [Ruff](https://docs.astral.sh/ruff/) linting/formatting and [deptry](https://deptry.com/) dependency validation.

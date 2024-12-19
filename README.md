

## Initial Setup
- Clone the repository
- Modify `pyproject.toml`.

## UV (if not installed)
- Install
    ```
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
- Path
    ```
    source .cargo/env
    ```

## Create environmen
- Create .venv
    ```
    uv sync
    ```
- Activate .venv
    ```
    . .venv/bin/activate
    ```
- Install pre-commit
    ```
    pre-commit install
    ```


## Singularity
- Check singularity.def
- Build
    ```
    singularity build --fakeroot singularity.sif singularity.def
    ```

## Docker
- Check Dockerfile
- Build
    ```
    docker build -t project:lastest .
    ```
- Run with `src` and `pyproject.toml` mounted
    ```
    docker run -it \
        -v $(pwd)/src:/app/src \
        -v $(pwd)/pyproject.toml:/app/pyproject.toml \
        -v $(pwd)/ruff.sh:/app/ruff.sh \
        project:lastest
    ```
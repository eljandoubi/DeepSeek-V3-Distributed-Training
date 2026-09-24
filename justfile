# Task runner for DeepSeek-V3 Distributed Training. Run `just` to list recipes.
# https://github.com/casey/just

set shell := ["bash", "-uc"]

# List available recipes
default:
    @just --list

# Install project + dev dependencies (ruff, pytest)
sync:
    uv sync --group dev

# Lint the codebase with ruff
lint:
    uv run ruff check .

# Auto-fix lint issues with ruff
fix:
    uv run ruff check --fix .

# Format the codebase with ruff
fmt:
    uv run ruff format .

# Check formatting without modifying files (as run in CI)
fmt-check:
    uv run ruff format --check .

# Run the same lint/format checks as CI
check: lint fmt-check

# Run the test suite (CPU-only, gloo backend)
test *args:
    uv run pytest tests/ -v {{ args }}

# Launch local training with a parallelism preset (see src/config/default_configs.py)
train config nproc="1":
    src_CONFIG={{ config }} uv run torchrun --nproc_per_node={{ nproc }} -m src.train

# Submit a multi-node training job via SLURM
slurm config nodes="2" gpus="8":
    src_CONFIG={{ config }} sbatch --nodes={{ nodes }} --gpus-per-node={{ gpus }} scripts/slurm_train.sbatch

# Remove caches and build artifacts
clean:
    rm -rf .pytest_cache .ruff_cache dist build
    find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +

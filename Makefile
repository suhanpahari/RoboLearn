# Roborto - common commands. See docs/PLAN.md.
.PHONY: help setup-gym setup-hab lint format test smoke train launch eval sweep figures clean

SHELL := /bin/bash
PYTHON ?= python
EXP ?= exp001_fetch_sac_her

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup-gym:  ## Create the manipulation conda env (Gymnasium-Robotics)
	conda env create -f environment-gymrobotics.yml

setup-hab:  ## Create the Habitat conda env
	conda env create -f environment-habitat.yml

lint:  ## Lint and format-check
	ruff check roborto scripts tests
	ruff format --check roborto scripts tests

format:  ## Auto-format and auto-fix
	ruff format roborto scripts tests
	ruff check --fix roborto scripts tests

test:  ## Run the unit tests
	$(PYTHON) -m pytest -q tests

smoke:  ## Tiny end-to-end run. Usage: make smoke EXP=expNNN GPU=GPU-xxxx
	@test -n "$(GPU)" || { echo "ERROR: set GPU=<uuid> (find an idle one: nvidia-smi -L)"; exit 1; }
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/train.py experiment=$(EXP) trainer.smoke=true

train:  ## Full training run. Usage: make train EXP=expNNN GPU=GPU-xxxx
	@test -n "$(GPU)" || { echo "ERROR: set GPU=<uuid> (find an idle one: nvidia-smi -L)"; exit 1; }
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/train.py experiment=$(EXP)

launch:  ## Launch all seeds in parallel. Usage: make launch EXP=expNNN GPUS=GPU-aa,GPU-bb,...
	@test -n "$(GPUS)" || { echo "ERROR: set GPUS=uuid1,uuid2,... (one per seed, find idle ones: nvidia-smi -L)"; exit 1; }
	$(PYTHON) scripts/launch_seeds.py experiment=$(EXP) gpus=$(GPUS)

eval:  ## Evaluate a run. Usage: make eval EXP=expNNN GPU=GPU-xxxx
	@test -n "$(GPU)" || { echo "ERROR: set GPU=<uuid> (find an idle one: nvidia-smi -L)"; exit 1; }
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/evaluate.py experiment=$(EXP)

sweep:  ## Print a sweep grid. Usage: make sweep SWEEP=her_relabel_strategy
	$(PYTHON) scripts/sweep.py sweep=$(SWEEP)

figures:  ## Regenerate all report figures from logged data
	$(PYTHON) scripts/make_figures.py

clean:  ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache

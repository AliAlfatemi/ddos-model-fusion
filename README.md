# DDoS Model Fusion

[![CI](https://github.com/AliAlfatemi/ddos-model-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/AliAlfatemi/ddos-model-fusion/actions/workflows/ci.yml)

PyTorch model definitions and score/rank fusion for aligned model predictions,
with a command-line interface and synthetic tests.

**Paper:** [Identifying Distributed Denial of Service Attacks Through Multi-Model Deep Learning Fusion and Combinatorial Analysis](https://doi.org/10.1007/s10922-024-09882-0)

**Authors:** Ali Alfatemi, Mohamed Rahouti, D. Frank Hsu, Christina Schweikert, Nasir Ghani, Aiman Solyman, Mohammad I. Saryuddin Assaqty

**Publication:** Journal of Network and Systems Management, volume 33, article 8 (2025).

## Install and validate

```bash
git clone https://github.com/AliAlfatemi/ddos-model-fusion.git
cd ddos-model-fusion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m ddos_fusion smoke
```

## Combine aligned predictions

Supply one CSV containing a unique `ID` and numeric score columns. Each row
must represent the same item across all model columns, and higher scores must
consistently mean greater positive-class confidence.

```bash
python -m ddos_fusion fuse --csv /path/to/aligned-predictions.csv \
  --columns model_a model_b model_c model_d \
  --weights 1 1 1 1 --method score --output outputs/fusion-001
```

Use `--method rank` for average-rank combination with tied ranks. Use
`--normalize` to min-max scale each score column within the supplied batch;
constant columns receive a neutral value of 0.5. Weights must be finite,
nonnegative, and have positive total. Choose weights using independent
validation data; the command does not learn weights from evaluation labels.
It writes fused scores/ranks and method metadata to a fresh local directory.

## Library components

- `ddos_fusion.models`: Conv1, Model2 (GRU), Model3 (MLP), and Model4 (residual network).
- `ddos_fusion.fusion`: validated score/rank combination for aligned arrays/data frames.
- `ddos_fusion.reference`: original fixed-column cognitive-diversity calculations.

The model classes provide building blocks for custom PyTorch training pipelines.
The CLI combines predictions from externally trained models. Reference diversity
functions use fixed model-column names and are available separately from the
CLI weighting interface.

## Implementation notes

See [implementation notes](docs/implementation.md) for model behavior, input
validation, and the fusion interface.

## Development, citation, and license

```bash
python -m pytest -q
python -m build
```

CI checks Python 3.12/3.13, model forward passes, fusion behavior, and the installed
wheel. [CITATION.cff](CITATION.cff) contains the complete published author order.
See [CONTRIBUTING.md](CONTRIBUTING.md) and [LICENSE](LICENSE).

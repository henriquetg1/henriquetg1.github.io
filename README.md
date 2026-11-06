# ANN & Deep Learning — Henrique Turco

Portfolio for the course Artificial Neural Networks and Deep Learning (Insper, 2026.2).

Published site: <https://henriquetg1.github.io/ann-dl/>

## Setup

```shell
python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r requirements.txt --upgrade
```

## Re-running the exercises

```shell
python docs/exercises/data/code/main.py
python docs/exercises/perceptron/code/main.py
python docs/exercises/mlp/code/exercises.py
```

## Site

```shell
mkdocs serve -o
```

The GitHub Actions workflow in `.github/workflows/main.yaml` publishes the site to
GitHub Pages (`gh-pages` branch) on every push to `main`.

# Artificial Neural Networks and Deep Learning

???+ info inline end "Edition"

    2026.2 — Insper

**Henrique Turco** — [github.com/henriquetg1](https://github.com/henriquetg1)

This site is my portfolio for the course
[Artificial Neural Networks and Deep Learning](https://insper.github.io/ann-dl/2026.2/)
(Insper, 2026.2). Each exercise lives in its own folder under `docs/exercises/`,
with the report (`index.md`), the code that was actually run (`code/`), and the
figures shown in the report (`figures/`).

## Deliverables

- [x] [1. Data](exercises/data/index.md) — 10.sep.2026
- [x] [2. Perceptron](exercises/perceptron/index.md) — 22.sep.2026
- [x] [3. MLP](exercises/mlp/index.md) — 06.nov.2026
- [x] [4. VAE](exercises/vae/index.md) — 06.nov.2026
- [ ] Projects

## Reproducing

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
python docs/exercises/data/code/main.py         # Data: figures, tables, results.json
python docs/exercises/perceptron/code/main.py   # Perceptron: figures, results.json
python docs/exercises/mlp/code/exercises.py     # MLP: figures, printed results
python docs/exercises/vae/code/exercises.py     # VAE: figures, printed results
mkdocs serve -o                           # preview the site locally
```

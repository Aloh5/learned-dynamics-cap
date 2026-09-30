# Current-paper numerical analysis

Run from the repository root:

```sh
python analysis/run_equilibria.py
python analysis/generate_attractors.py
python analysis/generate_candidates.py
python -m unittest discover -s analysis -p 'test_*.py' -v
```

The equilibrium routines are independent of the abandoned DDSR predictor comparison. They support Sections 2.1 and 2.2 of the current manuscript. See the root README for inputs, outputs, and environment requirements.

# Data to accompany "Clifford Circuit Optimization with Templates and Symbolic Pauli Gates"

Dependencies:

```
pip install -U qiskit==0.25.2 pandas seaborn tqdm matplotlib
```

To reproduce the figures and tables, uncompress `all-data.zip` into the folder containing `generate_figures_and_tables.py`. Then run `generate_figures_and_tables.py`:

```
unzip all-data.zip
python generate_figures_and_tables.py
```

Note the `generate_figures_and_tables.py` also verifies that all circuits are correct. This can take multiple hours. To skip, simply comment out the verification code (lines 44 and 103)  

# Generates figures and tables from the csv files
#
# dependencies:
#   - numpy
#   - matplotlib
#   - pandas (pip install pandas==1.1.2)
#   - seaborn
#   - Qiskit (pip install qiskit)
# To run: `python generate_figures_and_tables.py`

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from qiskit import QuantumCircuit
from qiskit.quantum_info import Clifford
import time

def verify_one_experiment(row, AG):
    c_orig = Clifford(QuantumCircuit.from_qasm_str(row['original']))
    c_comp = Clifford(QuantumCircuit.from_qasm_str(row['compiled']))
    c_opt = Clifford(QuantumCircuit.from_qasm_str(row['optimized']))
    assert(c_orig == c_comp)
    assert(c_comp == c_opt)
    if AG:
        c_opt_AG = Clifford(QuantumCircuit.from_qasm_str(row['compiled_aaronson_gottesman']))
        assert(c_opt == c_opt_AG)


def verify(df, AG=False):
    start = time.time()
    df.apply(lambda row: verify_one_experiment(row, AG), axis=1)
    end = time.time()
    print(f"Completed in {end - start:.1f} sec")
    print('All OK')

df = pd.read_csv("hamiltonian_evolution.csv")

# Should produce the same Clifford tableau

print("Verifying Hamiltonian evolution circuit results (this may take up to a few hours; comment out the next line to skip)")
verify(df, True)

# Generate LaTeX code for Table 1

from itertools import chain 

rows = []
for n in chain(range(5, 61, 10), [64]):
    dfc = df[(df['height'] == n) & (df['type'] == 'path')]
    improvement = (dfc['compiled_aaronson_gottesman_cost'].mean() - dfc['opt_cost'].mean()) / dfc['compiled_aaronson_gottesman_cost'].mean() * 100    
    print(f" Path graph & {int(dfc['nqubits'].mean())} & {len(dfc)} & {dfc['orig_cost'].mean():.2f} & {dfc['compiled_aaronson_gottesman_cost'].mean():.2f} & {dfc['compiled_cost'].mean():.2f} & {dfc['opt_cost'].mean():.2f} & {improvement:.2f} \\\\")

for n in chain(range(5, 61, 10), [64]):
    dfc = df[(df['height'] == n) & (df['type'] == 'cycle_graph')]
    improvement = (dfc['compiled_aaronson_gottesman_cost'].mean() - dfc['opt_cost'].mean()) / dfc['compiled_aaronson_gottesman_cost'].mean() * 100    
    print(f" Cycle graph & {int(dfc['nqubits'].mean())} & {len(dfc)} & {dfc['orig_cost'].mean():.2f}  & {dfc['compiled_aaronson_gottesman_cost'].mean():.2f}  & {dfc['compiled_cost'].mean():.2f} & {dfc['opt_cost'].mean():.2f} & {improvement:.2f} \\\\")
    
for n in range(3,9):
    dfc = df[(df['height'] == n) & (df['type'] == 'square_grid')]
    improvement = (dfc['compiled_aaronson_gottesman_cost'].mean() - dfc['opt_cost'].mean()) / dfc['compiled_aaronson_gottesman_cost'].mean() * 100    
    print(f" Square lattice & {int(dfc['nqubits'].mean())} & {len(dfc)} & {dfc['orig_cost'].mean():.2f}  & {dfc['compiled_aaronson_gottesman_cost'].mean():.2f}  & {dfc['compiled_cost'].mean():.2f} & {dfc['opt_cost'].mean():.2f} & {improvement:.2f} \\\\")
    
for n in range(2, 7):
    dfc = df[(df['height'] == n) & (df['type'] == 'triangular_lattice')]
    improvement = (dfc['compiled_aaronson_gottesman_cost'].mean() - dfc['opt_cost'].mean()) / dfc['compiled_aaronson_gottesman_cost'].mean() * 100    
    print(f" Triangular lattice & {int(dfc['nqubits'].mean())} & {len(dfc)} & {dfc['orig_cost'].mean():.2f}  & {dfc['compiled_aaronson_gottesman_cost'].mean():.2f}  & {dfc['compiled_cost'].mean():.2f} & {dfc['opt_cost'].mean():.2f} & {improvement:.2f} \\\\")

for n in range(1, 3):
    dfc = df[(df['height'] == n) & (df['type'] == 'hexagonal_lattice')]
    improvement = (dfc['compiled_aaronson_gottesman_cost'].mean() - dfc['opt_cost'].mean()) / dfc['compiled_aaronson_gottesman_cost'].mean() * 100    
    print(f" Hexagonal lattice & {int(dfc['nqubits'].mean())} & {len(dfc)} & {dfc['orig_cost'].mean():.2f}  & {dfc['compiled_aaronson_gottesman_cost'].mean():.2f}  & {dfc['compiled_cost'].mean():.2f} & {dfc['opt_cost'].mean():.2f} & {improvement:.2f} \\\\")

for (n1, n2) in [(2,1),(3,1),(2,2),(4,1),(3,2)]:
    dfc = df[(df['height'] == int(f'{n1}{n2}')) & (df['type'] == 'heavy_hex')]
    improvement = (dfc['compiled_aaronson_gottesman_cost'].mean() - dfc['opt_cost'].mean()) / dfc['compiled_aaronson_gottesman_cost'].mean() * 100    
    print(f" Heavy hex lattice & {int(dfc['nqubits'].mean())} & {len(dfc)} & {dfc['orig_cost'].mean():.2f}  & {dfc['compiled_aaronson_gottesman_cost'].mean():.2f}  & {dfc['compiled_cost'].mean():.2f} & {dfc['opt_cost'].mean():.2f} & {improvement:.2f} \\\\")

# Compute average improvement for Hamiltonian evolution

print(f"Average improvement over A-G: {(df['compiled_aaronson_gottesman_cost'].mean() - df['opt_cost'].mean()) / df['compiled_aaronson_gottesman_cost'].mean() * 100} %")
print(f"Average improvement over A-G (just greedy compiler): {(df['compiled_aaronson_gottesman_cost'].mean() - df['compiled_cost'].mean()) / df['compiled_aaronson_gottesman_cost'].mean() * 100} %")
print(f"Average improvement over A-G: {(df['compiled_aaronson_gottesman_cost'].mean()) / df['opt_cost'].mean()}")
print(f"Average improvement over A-G (just greedy compiler): {(df['compiled_aaronson_gottesman_cost'].mean()) / df['compiled_cost'].mean()}")

# Optimal n 6 circuits

timelims = [100, 139, 194, 270, 376, 524, 729, 1015, 1414, 1969, 2742, 3600, 5318, 7405, 10312, 14360, 19997, 27847, 36000, 54000]
dfs = {}
for i in timelims:
    df = pd.read_csv(f"optimal_n6_timelimit_{i}.csv")
    dfs[i] = df

df_all_res_in_one = pd.concat([dfs[i] for i in timelims], ignore_index=True)

print("Verifying results for optimal circuits on 6 qubits (this may take up to a few hours; comment out the next line to skip)")
verify(df_all_res_in_one, False)

df = dfs[14360]
costs = set(df['true_optimal_cost'])
print(f"Ratio of the circuits for which the optimal CNOT cost is recovered {100 * sum(df['recovered_optimal']) / len(df['recovered_optimal']):.2f}%")
print(f"Maximum overhead: {max(df['opt_cost']-df['true_optimal_cost'])} CNOT")
print(f"Average overhead: {((df['opt_cost']-df['true_optimal_cost']) / df['true_optimal_cost']).mean()*100:.2f}%")

from operator import itemgetter

overheads = []
total_over_time = 0
total_experiments = 0
for timelim, df in dfs.items():
    timediff = df['runtime'] - timelim
    total_over_time += sum(df['runtime'] > timelim)
    total_experiments += len(df)
    overheads.append(((sum(df['runtime'] > timelim) / len(df)) * 100, timelim))
print(f"Maximum ratio of the problems that did not complete in time due to letting the algorithm complete after time limit is exhausted: {max(overheads, key=itemgetter(0))[0]:.2f}%, reached on timelimit={max(overheads, key=itemgetter(0))[1]}")
print(f"Ratio of all: {(total_over_time / total_experiments) * 100:.2f}%")

# Generate plots in Fig. 2

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

# Constants
figwidth = 1.2 * (7.1413 / 2)
figheight = 1.2 * (figwidth / 1.618)
linewidth = 0.6
markersize = 3
fontsize = 10

rows = []
for timelim, df in dfs.items():
    costs = set(df['true_optimal_cost'])
    ratios = {'timelim': timelim}
    for cost in costs:
        recovered_for_cost_nq = df[df['true_optimal_cost'] == cost]['recovered_optimal']
        assert(len(recovered_for_cost_nq) == 100 or (len(recovered_for_cost_nq) == 3 and cost == 15))
        ratios[cost] = (sum(recovered_for_cost_nq)/ len(recovered_for_cost_nq))
    rows.append(ratios)

dfplt = pd.DataFrame(rows, columns=rows[0].keys()).set_index('timelim')

sns.set_style("whitegrid")
fig, ax = plt.subplots()
dfplt.plot(figsize = (figwidth,figheight), 
           xlabel='Time limit (s)', 
           ylabel='Ratio of circuits with optimal \# CNOTs', 
           marker='o',
           colormap=sns.color_palette("viridis", as_cmap=True),
           logx=True,
           fontsize=fontsize,
           linewidth=linewidth,
           markersize=markersize,
           ax=ax)
ax.legend(title='Optimal CNOT cost', ncol=2,prop={'size': 8})
plt.tight_layout()
plt.savefig("timelimt_sweep.pdf")

rows = []
for timelim, df in dfs.items():
    df['recovered_optimal'] = df.apply(lambda row: row['true_optimal_cost'] == row['opt_cost'], axis=1)

    costs = set(df['true_optimal_cost'])
    times = {'timelim': timelim}
    for cost in costs:
        times[cost] = df[df['true_optimal_cost'] == cost]['runtime'].mean()
    rows.append(times)

dfplt = pd.DataFrame(rows, columns=rows[0].keys()).set_index('timelim')

sns.set_style("whitegrid")
fig, ax = plt.subplots()
dfplt.plot(figsize = (figwidth,figheight),
           xlabel='Time limit (s)',
           ylabel='Mean running time (s)',
           marker='o',
           colormap=sns.color_palette("viridis", as_cmap=True),
           logy=True,
           logx=True,
           fontsize=fontsize,
           linewidth=linewidth,
           markersize=markersize,
           legend=False,
           ax=ax)

xmin = min(ax.get_xlim())
xmax = max(ax.get_xlim())
ymin = min(ax.get_ylim())
ymax = max(ax.get_ylim())

lims = [
    max([xmin, ymin]),
    min([xmax, ymax]),
]

# now plot both limits against eachother
ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

plt.tight_layout()
plt.savefig("timelime_sweep_runtimes.pdf")

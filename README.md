## Overview
This repo contains the codes for the manuscript "Sequence-Specific Minimizers via Polar Sets".
Polar set is a new way to create sequence-specific minimizers that overcomes several shortcomings in previous approaches to optimize a minimizer sketch specifically for a given reference sequence. For more details,
check out the preprint (to appear on bioRxiv).
This repo contains an implementation of our proposed Layered Polar Sets heuristics written in pure Python3, as well as codes to evaluate the heuristics and produce figures.

*Note: At the start of the development, the name "anchor sets" is used for the codename of the new framework. It was changed to "polar sets" later in the development process.*
## Preparation
### Dependencies
This repo is written in Pure Python and requires Python 3.6 or higher.
It requires the following packages: `bitarray, numpy, tqdm`, some for historical reasons.
### Data File Preparation
Open `anchor_distributed.py` and update the variable `working_dir` to where the results are intended to be stored.
Also ensure the directory `working_dir/preload` exists.
Download the sequence files and prepare a file with extension `.seq` that contains a single line for the whole sequence, in `ACGT` alphabet.
**Warning: Due to technical limitations, the sequence cannot exceed 2^32 characters.**
For demonostration purposes, we assume the file is named `TARGET.seq`.
### Preprocessing
```
python anchor_sets_bit.py TARGET
```
This generates the suffix array and the LCP array for the given sequence, and store the results at file `working_dir/preload/TARGET_30.dump`. Currently this runs rather slow, as we use an inefficient implementation.
This only needs to be run once for each sequence.
## Main Process
### Heuristics for Layered Polar Sets
```
python anchor_distributed.py TARGET -w 100 --kmin 15 --kmax 25 -p 10
```
The command above loads the preprocessed data for `TARGET.seq`, and spawns 10 subprocesses to run and evaluate the heuristics (and other algorithms), over the parameter range `w=100,15<=k<=25`.
*Note: Currently we only support k<=30.*
*Note: It is highly recommended to disable memory overcommitment protection if you intend to run more processes. In our implementation, different subprocesses share the preprocessed suffix arrays via Python global variables, which is copy-on-write. Even if the subprocesses will NEVER write to the suffix arrays, sufficient memory will be allocated for them, leading to early-than-expected out-of-memory complaints.*
### Hyperparameters
The list of algorithms to run can be found at the variable `anchor_distributed.py:fs_pairs`. The heuristics proposed and tested in the bioRxiv preprint is codenamed `multi_lax`, which is implemented at `anchor_ds_v3:AnchorStrategyV3:multi_random_pass`.
There are several hyperparameters for the heuristics. Most importantly, the number of rounds, the thresholds, and whether rounds are monotonic are implemented in the same `multi_random_pass` function.
### Runtime Inspection
The subprocesses will write logs to file `working_dir/{strategy_name}_TARGET_{w}_{k}.log`, which persists between runs. The reported density savings are measured using link energy, and the final reported density factors
are measured by emulation of a random compatible minimizer.
**Warning: While the program outputs a second value being the density factor on a random sequence, that value serves as a placeholder now and is retained for backwards compatibility.**
The `stdout` of the caller reports when tasks are completed, and how many tasks are left in the pool.
The file `working_dir/master_dat` contains the specific densities of optimized/evaluated methods and will be used for plotting.
## Evaluation
There are currently no unified interface for reproducing all plots in the manuscript, due to them being produced over a long period of time (some processes takes a very long time to complete). This will be fixed in a
future release. Currently, these are organized as parameterless top-level functions in the analysis code files, with self-explanatory names.
### Trying out Optimized Minimizers on other sequences
```
python real_seq_ops.py {various parameters}
```
The optimized layered anchor sets will be serialized to file `working_dir/{strategy_name}_TARGET_{w}_{k}.dump`. The file `real_seq_ops.py` provides interfaces to load these dump files (via class `LayeredAnchors`), and apply them to different sequences (function `calc_energy_stats` to calculate energy surplus and deficit, function `calc_energy` to calculate expected density, function `calc_selected_locs` for actually instantiation of random compatible minimizers, which also provides support for Robust Winnowing). It also provides reference implementation of random minimizers and the Miniception, serving as alternative algorithms to compare against.
Currently, one needs to uncomment different lines in the main body to run different subroutines, although each of them are pretty self-explanatory.
### Generating Plots
```
python plot.py
```
With correct datas parsed (this includes energy surplus and deficit for sequences, and the performance of different minimizers on random sequences as well as target sequences), this generates all six figures in the
current manuscript. Currently the progress plots are parsed from the log files of optimization routines, and all other plots fetch data from `working_dir/master_dat` and other `.dat` files generated by the real sequence
related processes.
The primary result plots are generated using functions in file `plot.py`. Due to the structure of our experiments, we usually make two plots in one file while sharing legend between them. This is implemented by the
function `parallel_plot`.

---
title: iGedi
emoji: 🌖
colorFrom: indigo
colorTo: pink
sdk: streamlit
sdk_version: 1.38.0
python_version: "3.12"
app_file: utils/config_fabric.py
pinned: false
license: mit
---

<p>
  <img src="gedi/utils/logo.png" alt="Logo" width="100" align="left" />
  <h1 style="display: inline;">(i)GEDI</h1>
</p>

(**i**nteractive) **G**enerating **E**vent **D**ata with **I**ntentional Features for Benchmarking Process Mining<br />
This repository contains the codebase for the interactive web application tool (iGEDI) as well as for the [GEDI paper](https://mcml.ai/publications/gedi.pdf) accepted at the BPM'24 conference.

## Table of Contents

- [Interactive Web Application (iGEDI)](#interactive-web-application)
- [Requirements](#requirements)
- [Installation](#installation)
- [General Usage](#general-usage)
- [Experiments](#experiments)
- [Citation](#citation)

## Interactive Web Application
Our [interactive web application](https://huggingface.co/spaces/andreamalhera/gedi) (iGEDI) guides you through the specification process, runs GEDI for you. You can directly download the resulting generated logs or the configuration file to run GEDI locally.
![Interface Screenshot](gedi/utils/iGEDI_interface.png)

## Requirements
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Graphviz on your OS e.g.
For MacOS:
```console
brew install graphviz
brew install swig
```
- For smac:
```console
conda install pyrfr swig
```

## Installation
To directly use GEDI methods via `import gedi`, install directly from [PyPi](https://pypi.org/project/gedi/).
```console
pip install gedi
```
Alternatively, you can create an environment with
- `conda env create -f .conda.yml`

Run:
```console
python -c "from gedi import gedi; gedi('config_files/pipeline_steps/generation.json')"
```
or

```console
conda activate gedi
python main.py -a config_files/test/experiment_test.json
```
The last step should take only a few minutes to run.

## General Usage
Our pipeline offers several pipeline steps, which can be run sequentially or partially ordered:
- [Feature Extraction](#feature-extraction)
- [Generation](#generation)
- [Benchmark](#benchmark)
- [Evaluation Plotter](https://github.com/andreamalhera/gedi/blob/16-documentation-update-readme/README.md#evaluation-plotting)

To run different steps of the GEDI pipeline, please adapt the `.json` accordingly.
```console
conda activate gedi
python main.py -a config_files/pipeline_steps/<pipeline-step>.json
```
For reference of possible keys and values for each step, please see `config_files/test/experiment_test.json`.
To run the whole pipeline please create a new `.json` file, specifying all steps you want to run and specify desired keys and values for each step.
To reproduce results from our paper, please refer to [Experiments](#experiments).

### Feature Extraction
---
To extract the features on the event-log level and use them for hyperparameter optimization, we employ the following script:
```console
conda activate gedi
python main.py -a config_files/pipeline_steps/feature_extraction.json
```
The JSON file consists of the following key-value pairs:

- pipeline_step: denotes the current step in the pipeline (here: feature_extraction)
- input_path: folder to the input files
- feature params: defines a dictionary, where the inner dictionary consists of a key-value pair 'feature_set' with a list of features being extracted from the references files. A list of valid features can be looked up from the FEEED extractor
- output_path: defines the path, where plots are saved to
- real_eventlog_path: defines the file with the features extracted from the real event logs
- plot_type: defines the style of the output plotting (possible values: violinplot, boxplot)
- font_size: label font size of the output plot
- boxplot_width: width of the violinplot/boxplot

### Generation
---
After having extracted meta features from the files, the next step is to generate event log data accordingly. Generally, there are two settings on how the targets are defined: i) meta feature targets are defined by the meta features from the real event log data; ii) a configuration space is defined which resembles the feasible meta features space. 

The command to execute the generation step is given by a exemplarily generation.json file:

```console
conda activate gedi
python main.py -a config_files/pipeline_steps/generation.json
```

In the `generation.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: event_logs_generation)
* targets: defines the path to the input file which contains the features that are used for the optimization step. The 'objectives' define the specific features, which are the optimization criteria.
* config_space: here, we define the configuration of the generator module (here: process tree generator). The process tree generator can process input information which defines characteristics for the generated data (a more thorough overview of the params can be found [here](https://github.com/tjouck/PTandLogGenerator):

        - mode: most frequent number of visible activities
        - sequence: the probability of adding a sequence operator to the tree
        - choice: the probability of adding a choice operator to the tree
        - parallel: the probability of adding a parallel operator to the tree
        - loop: the probability of adding a loop operator to the tree
        - silent: probability to add silent activity to a choice or loop operator
        - lt_dependency: the probability of adding a random dependency to the tree
        - num_traces: the number of traces in the event log
        - duplicate: the probability of duplicating an activity label
        - or: probability to add an or operator to the tree

* system_params:
    - output_path: defines the output folder
    - n_trials: the maximum number of trials for the hyperparameter optimization to find a feasible solution to the specific configuration being used as the target
* plot_reference_feature: defines the feature, which is used on the x-axis on the output plots, i.e., each feature defined in the 'objectives' of the 'target' is plotted against the reference feature being defined in this value

In case of manually defining the targets for the features in config space, the following table shows the range of the features in the real-world event log data (BPIC's) for reference:
<div style="overflow-x:auto;">
    <table border="1" class="dataframe">
    <thead>
        <tr style="text-align: right;">
        <th></th>
        <th>n_traces</th>
        <th>n_variants</th>
        <th>ratio_variants_per_number_of_traces</th>
        <th>trace_len_min</th>
        <th>trace_len_max</th>
        <th>trace_len_mean</th>
        <th>trace_len_median</th>
        <th>trace_len_mode</th>
        <th>trace_len_std</th>
        <th>trace_len_variance</th>
        <th>trace_len_q1</th>
        <th>trace_len_q3</th>
        <th>trace_len_iqr</th>
        <th>trace_len_geometric_mean</th>
        <th>trace_len_geometric_std</th>
        <th>trace_len_harmonic_mean</th>
        <th>trace_len_skewness</th>
        <th>trace_len_kurtosis</th>
        <th>trace_len_coefficient_variation</th>
        <th>trace_len_entropy</th>
        <th>trace_len_hist1</th>
        <th>trace_len_hist2</th>
        <th>trace_len_hist3</th>
        <th>trace_len_hist4</th>
        <th>trace_len_hist5</th>
        <th>trace_len_hist6</th>
        <th>trace_len_hist7</th>
        <th>trace_len_hist8</th>
        <th>trace_len_hist9</th>
        <th>trace_len_hist10</th>
        <th>trace_len_skewness_hist</th>
        <th>trace_len_kurtosis_hist</th>
        <th>ratio_most_common_variant</th>
        <th>ratio_top_1_variants</th>
        <th>ratio_top_5_variants</th>
        <th>ratio_top_10_variants</th>
        <th>ratio_top_20_variants</th>
        <th>ratio_top_50_variants</th>
        <th>ratio_top_75_variants</th>
        <th>mean_variant_occurrence</th>
        <th>std_variant_occurrence</th>
        <th>skewness_variant_occurrence</th>
        <th>kurtosis_variant_occurrence</th>
        <th>n_unique_activities</th>
        <th>activities_min</th>
        <th>activities_max</th>
        <th>activities_mean</th>
        <th>activities_median</th>
        <th>activities_std</th>
        <th>activities_variance</th>
        <th>activities_q1</th>
        <th>activities_q3</th>
        <th>activities_iqr</th>
        <th>activities_skewness</th>
        <th>activities_kurtosis</th>
        <th>n_unique_start_activities</th>
        <th>start_activities_min</th>
        <th>start_activities_max</th>
        <th>start_activities_mean</th>
        <th>start_activities_median</th>
        <th>start_activities_std</th>
        <th>start_activities_variance</th>
        <th>start_activities_q1</th>
        <th>start_activities_q3</th>
        <th>start_activities_iqr</th>
        <th>start_activities_skewness</th>
        <th>start_activities_kurtosis</th>
        <th>n_unique_end_activities</th>
        <th>end_activities_min</th>
        <th>end_activities_max</th>
        <th>end_activities_mean</th>
        <th>end_activities_median</th>
        <th>end_activities_std</th>
        <th>end_activities_variance</th>
        <th>end_activities_q1</th>
        <th>end_activities_q3</th>
        <th>end_activities_iqr</th>
        <th>end_activities_skewness</th>
        <th>end_activities_kurtosis</th>
        <th>eventropy_trace</th>
        <th>eventropy_prefix</th>
        <th>eventropy_global_block</th>
        <th>eventropy_lempel_ziv</th>
        <th>eventropy_k_block_diff_1</th>
        <th>eventropy_k_block_diff_3</th>
        <th>eventropy_k_block_diff_5</th>
        <th>eventropy_k_block_ratio_1</th>
        <th>eventropy_k_block_ratio_3</th>
        <th>eventropy_k_block_ratio_5</th>
        <th>eventropy_knn_3</th>
        <th>eventropy_knn_5</th>
        <th>eventropy_knn_7</th>
        <th>epa_variant_entropy</th>
        <th>epa_normalized_variant_entropy</th>
        <th>epa_sequence_entropy</th>
        <th>epa_normalized_sequence_entropy</th>
        <th>epa_sequence_entropy_linear_forgetting</th>
        <th>epa_normalized_sequence_entropy_linear_forgetting</th>
        <th>epa_sequence_entropy_exponential_forgetting</th>
        <th>epa_normalized_sequence_entropy_exponential_forgetting</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td>[ min, max ]</td>
        <td>[ 226.0, 251734.0 ]</td>
        <td>[ 6.0, 28457.0 ]</td>
        <td>[ 0.0, 1.0 ]</td>
        <td>[ 1.0, 24.0 ]</td>
        <td>[ 1.0, 2973.0 ]</td>
        <td>[ 1.0, 131.49 ]</td>
        <td>[ 1.0, 55.0 ]</td>
        <td>[ 1.0, 61.0 ]</td>
        <td>[ 0.0, 202.53 ]</td>
        <td>[ 0.0, 41017.89 ]</td>
        <td>[ 1.0, 44.0 ]</td>
        <td>[ 1.0, 169.0 ]</td>
        <td>[ 0.0, 161.0 ]</td>
        <td>[ 1.0, 53.78 ]</td>
        <td>[ 1.0, 5.65 ]</td>
        <td>[ 1.0, 51.65 ]</td>
        <td>[ -0.58, 111.97 ]</td>
        <td>[ -0.97, 14006.75 ]</td>
        <td>[ 0.0, 4.74 ]</td>
        <td>[ 5.33, 12.04 ]</td>
        <td>[ 0.0, 1.99 ]</td>
        <td>[ 0.0, 0.42 ]</td>
        <td>[ 0.0, 0.4 ]</td>
        <td>[ 0.0, 0.19 ]</td>
        <td>[ 0.0, 0.14 ]</td>
        <td>[ 0.0, 10.0 ]</td>
        <td>[ 0.0, 0.02 ]</td>
        <td>[ 0.0, 0.04 ]</td>
        <td>[ 0.0, 0.0 ]</td>
        <td>[ 0.0, 2.7 ]</td>
        <td>[ -0.58, 111.97 ]</td>
        <td>[ -0.97, 14006.75 ]</td>
        <td>[ 0.0, 0.79 ]</td>
        <td>[ 0.0, 0.87 ]</td>
        <td>[ 0.0, 0.98 ]</td>
        <td>[ 0.0, 0.99 ]</td>
        <td>[ 0.2, 1.0 ]</td>
        <td>[ 0.5, 1.0 ]</td>
        <td>[ 0.75, 1.0 ]</td>
        <td>[ 1.0, 24500.67 ]</td>
        <td>[ 0.04, 42344.04 ]</td>
        <td>[ 1.54, 64.77 ]</td>
        <td>[ 0.66, 5083.46 ]</td>
        <td>[ 1.0, 1152.0 ]</td>
        <td>[ 1.0, 66058.0 ]</td>
        <td>[ 34.0, 466141.0 ]</td>
        <td>[ 4.13, 66058.0 ]</td>
        <td>[ 2.0, 66058.0 ]</td>
        <td>[ 0.0, 120522.25 ]</td>
        <td>[ 0.0, 14525612122.34 ]</td>
        <td>[ 1.0, 66058.0 ]</td>
        <td>[ 4.0, 79860.0 ]</td>
        <td>[ 0.0, 77290.0 ]</td>
        <td>[ -0.06, 15.21 ]</td>
        <td>[ -1.5, 315.84 ]</td>
        <td>[ 1.0, 809.0 ]</td>
        <td>[ 1.0, 150370.0 ]</td>
        <td>[ 27.0, 199867.0 ]</td>
        <td>[ 3.7, 150370.0 ]</td>
        <td>[ 1.0, 150370.0 ]</td>
        <td>[ 0.0, 65387.49 ]</td>
        <td>[ 0.0, 4275524278.19 ]</td>
        <td>[ 1.0, 150370.0 ]</td>
        <td>[ 4.0, 150370.0 ]</td>
        <td>[ 0.0, 23387.25 ]</td>
        <td>[ 0.0, 9.3 ]</td>
        <td>[ -2.0, 101.82 ]</td>
        <td>[ 1.0, 757.0 ]</td>
        <td>[ 1.0, 16653.0 ]</td>
        <td>[ 28.0, 181328.0 ]</td>
        <td>[ 3.53, 24500.67 ]</td>
        <td>[ 1.0, 16653.0 ]</td>
        <td>[ 0.0, 42344.04 ]</td>
        <td>[ 0.0, 1793017566.89 ]</td>
        <td>[ 1.0, 16653.0 ]</td>
        <td>[ 3.0, 39876.0 ]</td>
        <td>[ 0.0, 39766.0 ]</td>
        <td>[ -0.7, 13.82 ]</td>
        <td>[ -2.0, 255.39 ]</td>
        <td>[ 0.0, 13.36 ]</td>
        <td>[ 0.0, 16.77 ]</td>
        <td>[ 0.0, 24.71 ]</td>
        <td>[ 0.0, 685.0 ]</td>
        <td>[ -328.0, 962.0 ]</td>
        <td>[ 0.0, 871.0 ]</td>
        <td>[ 0.0, 881.0 ]</td>
        <td>[ 0.0, 935.0 ]</td>
        <td>[ 0.0, 7.11 ]</td>
        <td>[ 0.0, 7.11 ]</td>
        <td>[ 0.0, 8.93 ]</td>
        <td>[ 0.0, 648.0 ]</td>
        <td>[ 0.0, 618.0 ]</td>
        <td>[ 0.0, 11563842.15 ]</td>
        <td>[ 0.0, 0.9 ]</td>
        <td>[ 0.0, 21146257.12 ]</td>
        <td>[ 0.0, 0.76 ]</td>
        <td>[ 0.0, 14140225.9 ]</td>
        <td>[ 0.0, 0.42 ]</td>
        <td>[ 0.0, 15576076.83 ]</td>
        <td>[ 0.0, 0.51 ]</td>
        </tr>
    </tbody>
    </table>
</div>

### Benchmark
The benchmarking defines the downstream task which is used for evaluating the goodness of the synthesized event log datasets with the metrics of real-world datasets. The command to execute a benchmarking is shown in the following script:

```console
conda activate gedi
python main.py -a config_files/pipeline_steps/benchmark.json
```

In the `benchmark.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: benchmark_test)
* benchmark_test: defines the downstream task. Currently (in v 1.0), only `discovery` for process discovery is implemented
* input_path: defines the input folder where the synthesized event log data are stored
* output_path: defines the output folder
* miners: defines the miners for the downstream task 'discovery' which are used in the benchmarking. In v 1.0 the miners 'inductive' for inductive miner, 'heuristics' for heuristics miner, 'imf' for inductive miner infrequent, as well as 'ilp' for integer linear programming are implemented


### Evaluation Plotting
The purpose of the evaluation plotting step is used just for visualization. Some examples of how the plotter can be used is shown in the following exemplarily script:


```console
conda activate gedi
python main.py -a config_files/pipeline_steps/evaluation_plotter.json
```

Generally, in the `evaluation_plotter.json`, we have the following key-value pairs:

* pipeline_step: denotes the current step in the pipeline (here: evaluation_plotter)
* input_path: defines the input file or the input folder which is considered for the visualizations. If a single file is specified, only the features in that file are considered whereas in the case of specifying a folder, the framework iterates over all files and uses them for plotting
* plot_reference_feature: defines the feature that is used on the x-axis on the output plots, i.e., each feature defined in the input file is plotted against the reference feature being defined in this value
* targets: defines the target values which are also used as reference. Likewise to the input_path, the targets can be specified by a single file or by a folder
* output_path: defines where to store the plots

## Experiments
In this repository, targets can be run selectively or from scratch, as preferred. For this purpose, we linked both inputs and outputs for each stage. In this section, we present the reproduction of generated event data, as in our paper, as well as the [visualization of evaluation figures](#visualizations).
We present two settings for generating intentional event logs, using [real targets](#generating-data-with-real-targets) or using [grid targets](#generating-data-with-grid-targets). Both settings output `.xes` event logs, `.json` and `.csv` files containing feature values, as well as evaluation results, from running a [process discovery benchmark](#benchmark), for the generated event logs.

### Generating data with real targets
To execute the experiments with real targets, we employ the [experiment_real_targets.json](config_files/experiment_real_targets.json). The script's pipeline will output the [generated event logs (GenBaselineED)](data/event_logs/GenBaselineED), which optimize their feature values towards [real-world event data features](data/BaselineED_feat.csv), alongside their respectively measured [feature values](data/GenBaselineED_feat.csv) and [benchmark metrics values](data/GenBaselineED_bench.csv).

```console
conda activate gedi
python main.py -a config_files/experiment_real_targets.json
```

### Generating data with grid targets
To execute the experiments with grid targets, a single [configuration](config_files/grid_2obj) can be selected or all [grid objectives](data/grid_2obj) can be run with one command using the following script. This script will output the [generated event logs (GenED)](data/event_logs/GenED), alongside their respectively measured [feature values](data/GenED_feat.csv) and [benchmark metrics values](data/GenED_bench.csv).
```
conda activate gedi
python gedi/utils/execute_grid_experiments.py config_files/test
```
We employ the [Config files fabric](notebooks/grid_experiment_configfiles_fabric.ipynb) to create all necessary [configuration](config_files/grid_2obj) and [objective](data/grid_2obj) files for this experiment.
For more details about these config_files, please refer to [Feature Extraction](#feature-extraction), [Generation](#generation), and [Benchmark](#benchmark).
To create configuration files for grid objectives interactively, you can use the start the following dashboard:
```
streamlit run utils/config_fabric.py # To tunnel to local machine add: --server.port 8501 --server.headless true

# In local machine (only in case you are tunneling):
ssh -N -f -L 9000:localhost:8501 <user@remote_machine.com>
open "http://localhost:9000/"
```

### Visualizations
To run the visualizations, we employ [jupyter notebooks](https://jupyter.org/install) and [add the installed environment to the jupyter notebook](https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084). We then start all visualizations by running e.g.: `jupyter noteboook`. In the following, we describe the `.ipynb`-files in the folder `\notebooks` to reproduce the figures from our paper. 

#### [Fig. 4 and fig. 5 Representativeness](notebooks/gedi_figs4and5_representativeness.ipynb)
To visualize the coverage of the feasible feature space of generated event logs compared to existing real-world benchmark datasets, in this notebook, we conduct a principal component analysis on the features of both settings. The first two principal components are utilized to visualize the coverage which is further highlighted by computing a convex hull of the 2D mapping.Additionally, we visualize the distribution of each meta feature we used in the paper as a boxplot. Additional features can be extracted with FEEED. Therefore, the notebook contains the figures 4 and 5 in the paper.

#### [Fig. 6 Benchmark Boxplots](notebooks/gedi_fig6_benchmark_boxplots.ipynb)
This notebook is used to visualize the metric distribution of real event logs compared to the generated ones. It shows 5 different metrics on 3 various process discovery techniques. We use 'fitness,', 'precision', 'fscore', 'size', 'cfc' (control-flow complexity) as metrics and as 'heuristic miner', 'ilp' (integer linear programming), and 'imf' (inductive miner infrequent) as miners. The notebook outputs the visualization shown in Fig.6 in the paper.

#### [Fig. 7 and fig. 8 Benchmark's Statistical Tests](notebooks/gedi_figs7and8_benchmarking_statisticalTests.ipynb)

This notebook is used to answer the question if there is a statistically significant relation between feature similarity and performance metrics for the downstream tasks of process discovery. For that, we compute the pearson coefficient, as well as the kendall's tau coefficient. This elucidates the correlation between the features with metric scores being used for process discovery. Each coefficient is calculated for three different settings: i) real-world datasets; ii) synthesized event log data with real-world targets; iii) synthesized event log data with grid objectives. Figures 7 and 8 shown in the paper refer to this notebook.

#### [Fig. 9 Consistency and fig. 10 Limitations](notebooks/gedi_figs9and10_consistency.ipynb)
Likewise to the evaluation on the statistical tests in notebook `gedi_figs7and8_benchmarking_statisticalTests.ipynb`, this notebook is used to compute the differences between two correlation matrices $\Delta C = C_1 - C_2$. This logic is employed to evaluate and visualize the distance of two correlation matrices. Furthermore, we show how significant scores are retained from the correlations being evaluated on real-world datasets coompared to synthesized event log datasets with real-world targets. In Fig. 9 and 10 in the paper, the results of the notebook are shown. 

## Citation
The `GEDI` framework is taken directly from the original paper by [Maldonado](mailto:andreamalher.works@gmail.com), Frey, Tavares, Rehwald and Seidl on BPM'24.

```bibtex
@InProceedings{maldonado2024gedi,
author="Maldonado, Andrea
and Frey, Christian M. M.
and Tavares, Gabriel Marques
and Rehwald, Nikolina
and Seidl, Thomas",
editor="Marrella, Andrea
and Resinas, Manuel
and Jans, Mieke
and Rosemann, Michael",
title="GEDI: Generating Event Data with Intentional Features for Benchmarking Process Mining",
booktitle="Business Process Management",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="221--237",
abstract="Process mining solutions include enhancing performance, conserving resources, and alleviating bottlenecks in organizational contexts. However, as in other data mining fields, success hinges on data quality and availability. Existing analyses for process mining solutions lack diverse and ample data for rigorous testing, hindering insights' generalization. To address this, we propose Generating Event Data with Intentional features, a framework producing event data sets satisfying specific meta-features. Considering the meta-feature space that defines feasible event logs, we observe that existing real-world datasets describe only local areas within the overall space. Hence, our framework aims at providing the capability to generate an event data benchmark, which covers unexplored regions. Therefore, our approach leverages a discretization of the meta-feature space to steer generated data towards regions, where a combination of meta-features is not met yet by existing benchmark datasets. Providing a comprehensive data pool enriches process mining analyses, enables methods to capture a wider range of real-world scenarios, and improves evaluation quality. Moreover, it empowers analysts to uncover correlations between meta-features and evaluation metrics, enhancing explainability and solution effectiveness. targets demonstrate GEDI's ability to produce a benchmark of intentional event data sets and robust analyses for process mining tasks.",
isbn="978-3-031-70396-6"
}
```
Furthermore, the `iGEDI` web application is taken directly from the original paper by [Maldonado](mailto:andreamalher.works@gmail.com), Aryasomayajula, Frey, and Seidl and is *to appear on Demos@ICPM'24*.
```bibtex
@inproceedings{maldonado2024igedi,
  author       = {Andrea Maldonado and
                  Sai Anirudh Aryasomayajula and
                  Christian M. M. Frey and
                  Thomas Seidl},
  editor       = {Jochen De Weerdt, Giovanni Meroni, Han van der Aa, and Karolin Winter},
  title        = {iGEDI: interactive Generating Event Data with Intentional Features},
  booktitle    = {ICPM 2024 Tool Demonstration Track, October 14-18, 2024, Kongens Lyngby, Denmark},
  series       = {{CEUR} Workshop Proceedings},
  publisher    = {CEUR-WS.org},
  year         = {2024},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

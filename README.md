# Benchmarking Detectors in SML

![animation](manim/animation.gif)



Optional project of the [Streaming Data Analytics](http://emanueledellavalle.org/teaching/streaming-data-analytics-2023-24/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=811164&polij_device_category=DESKTOP&__pj0=0&__pj1=d563c55e73c3035baf5b0bab2dda086b).

Student: **Tommaso Crippa**

## Background
Concept Drift occurs when the distribution of data changes over time, making a previously trained model less effective. In **Streaming Machine Learning** (SML) scenarios, it is critical to detect these changes early to adapt the model and maintain high performance. CapyMOA implements several Concept Drift Detectors, tools that help identify concept drifts to help models in the adaption. 

## Goals and objectives
The objective of this project is to conduct a benchmarking of different **Concept Drift Detectors** in CapyMOA, finding out their effectiveness given various evaluation metrics like accuracy, Kappa Statistic, runtime, and memory on multiple datasets. 

The end goal is to identify the pros and cons of each detector, and selecting which are the most useful in each scenario.

## Repository Structure

The repository is structured in the following way

- **notebooks/**  
  Jupyter and Colab-ready notebooks for evaluating and benchmarking drift detectors.

- **results/**  
  CSV files containing saved results from evaluations.

- **src/**  
  Source code and scripts:  
  - `evaluation.py` — Evaluates a single detector on a specified data stream.  
  - `benchmark_all.sh` — Runs evaluations across all datasets and detectors.  
  - Additional structures for managing experiments and evaluations.

- **manim/**  
  Code and gif utilized for initial animation

## Usage


After setting up the environment (check [**INSTALL.md**](INSTALL.md) for details), you can run your own experiments using the ```src/evaluation.py``` python script by specifying the following parameters.

```--stream```: Select a stream between Covtype, Electricity, Hyper100k, or Sensor.

```--classifier```: Select a classifier between NaiveBayes or HoeffdingTree.

```--detector```: Select a detector between ADWIN, STEPD, CUSUM, PageHinkley, DDM, HDDMAverage, HDDMWeighted or None.

#### Additional parameters

```--window_size```: Evaluation window size as a percentage (default: 1.0).

```--cooldown_window```: Cooldown window length (default: 0).

```--print_results```: Whether to print results to console (default: False).

```--save_results```: Whether to save results to CSV (default: False).

```--filename```: Output filename (default: "results.csv").

Example:
```bash
python src/evaluation.py \
    --stream Electricity \
    --classifier HoeffdingTree \
    --detector ADWIN \
    --window_size 1.0 \
    --cooldown_window 50 \
    --print_results True \
    --save_results True \
    --filename electricity_adwin_results.csv
```

To run all evaluations together, use the provided script:

```bash
./src/benchmark_all.sh
```

This will iterate through the combinations of streams, classifiers, and detectors, and save the aggregated results inside the results.csv file.

# 2024-2025 Benchmarking Detectors in SML

Optional project of the [Streaming Data Analytics](http://emanueledellavalle.org/teaching/streaming-data-analytics-2023-24/) course provided by [Politecnico di Milano](https://www11.ceda.polimi.it/schedaincarico/schedaincarico/controller/scheda_pubblica/SchedaPublic.do?&evn_default=evento&c_classe=811164&polij_device_category=DESKTOP&__pj0=0&__pj1=d563c55e73c3035baf5b0bab2dda086b).

Student: **Tommaso Crippa**

## Background
Concept Drift occurs when the distribution of data changes over time, making a previously trained model less effective. In Streaming Machine Learning (SML) scenarios, it is critical to detect these changes early to adapt the model and maintain high performance. CapyMOA implements several Concept Drift Detectors, tools that help identify concept drifts to help models in the adaption. 

The objective of this project is to conduct a benchmarking of different Concept Drift Detectors in CapyMOA, evaluating their impact on the performance of a baseline classifier.

## Goals and objectives
The project includes the following objectives:
- Test a Baseline Classifier without any drift detectors to measure its initial performance in terms of accuracy, execution time, and memory consumption.
- Compare the results obtained by applying different Concept Drift Detectors.
- Use different classic Streaming Machine Learning datasets to ensure a fair and meaningful comparison.
- Analyze how the integration of a Concept Drift Detector changes:
  - Accuracy/F1-score of the model.
  - Execution time required.
  - Memory and CPU utilization.
- Identify the most effective detector based on the conditions of each dataset.

## Datasets
The datasets used for benchmarking will include classic Streaming Machine Learning data, such as:
- *Electricity*: it is a classification problem based on the Australian New South Wales Electricity Market.
- *Covtype*: it contains the forest cover type for 30 x 30 meter cells obtained from US Forest Service (USFS) Region 2 Resource Information System (RIS) data.
- *Hyper100k*: it is a classification problem based on the moving hyperplane generator.
- *Sensor*: it is a classification problem based on indoor sensor data.

## Methodologies/models to apply
The methodologies and models used will be:
- *Baseline Classifier*: use both Hoeffding Tree and Naive Bayes to establish initial performance without drift detection.

Concept Drift Detectors:
- *ADWIN* (ADaptive WINdowing)
- *STEPD* (Statistical Test of Equal Proportions Drift)
- *CUSUM* (CUmulative SUM)
- *Page-Hinkley*
- *DDM* (Drift Detection Method)
- *HDDM_A* (Hoeffding Drift Detection Method - Absolute)
- *HDDM_W* (Hoeffding Drift Detection Method - Weighted)

## Evaluation metrics
The evaluation metrics used for comparison will be:
- *Accuracy/F1-score*: measuring the predictive performance of the model.
- *Execution time*: analysis of the time required to process the data stream with and without drift detection.
- *Memory Utilization*: amount of RAM required during execution.
- *CPU Utilization*: computational load generated.

## Deliverable
At the end of the project, the student must deliver:
- Notebooks with:
  - Implementation of Baseline Classifier without drift detection.
  - Comparison of performance with different Concept Drift Detectors.
  - Visualization of results with graphs and statistics.
  - Conclusions and recommendations on which detector performs best based on different datasets and different metrics.
    
This project will provide a detailed overview of Concept Drift Detectors in CapyMOA, highlighting their advantages, limitations and impact on the performance of Streaming Machine Learning models.

## Note for Students
- Clone the created repository offline;
- Add your name and surname into the Readme file;
- Make any changes to your repository, according to the specific assignment;
- Add a requirement.txt file for code reproducibility and instructions on how to replicate the results;
- Commit your changes to your local repository;
- Push your changes to your online repository.

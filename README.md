# GCAD 
This repository contains the official code for the AAAI-25 paper: <br>
*GCAD: Anomaly Detection in Multivariate Time Series from the Perspective of Granger Causality* <br>
In this paper, we designed a framework that models spatial dependencies using interpretable causal relationships and detects anomalies through **changes in causal patterns**. Specifically, we propose a method to dynamically discover Granger causality effects using gradients in nonlinear deep predictors and employ a simple sparsification strategy to obtain a Granger causality graph, detecting anomalies from a causal perspective. <br>
![image](https://github.com/Tc99m/GCAD/blob/master/img/model.png)
# Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.
- pytorch-tsmixer (https://github.com/ditschuk/pytorch-tsmixer)

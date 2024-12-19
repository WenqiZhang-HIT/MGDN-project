# Anomaly Detection in Industrial Control Systems Based on Cross-Domain Representation Learning

This project is based on the research presented in the paper **"Anomaly Detection in Industrial Control Systems Based on Cross-Domain Representation Learning"**, which has been accepted for publication in the **IEEE Transactions on Dependable and Secure Computing (TDSC)** in December 2024.

## Overview

**Industrial control systems (ICSs) are widely used** **in industry, and their security and stability are very important.** **Once the ICS is attacked, it may cause serious damage. Therefore,** **it is very important to detect anomalies in ICSs. ICS can monitor** **and manage physical devices remotely using communication** **networks. The existing anomaly detection approaches mainly** **focus on analyzing the security of network traffic or sensor data.** **However, the behaviors of different domains (e.g., network traffic** **and sensor physical status) of ICSs are correlated, so it is difficult** **to comprehensively identify anomalies by analyzing only a single** **domain. In this paper, an anomaly detection approach based** **on cross-domain representation learning in ICSs is proposed,** **which can learn the joint features of multi-domain behaviors and** **detect anomalies within different domains. After constructing across-domain graph that can represent the behaviors of multiple** **domains in ICSs, our approach can learn the joint features** **of them by leveraging graph neural networks. Since anomalies** **behave differently in different domains, we leverage a multi** **task learning approach to identify anomalies in different domains** **separately and perform joint training. The experimental results** **show that the performance of our approach is better than existing** **approaches for identifying anomalies in ICSs.**

## Installation

To install and run this project, follow these steps: 

### Prerequisites 

- **PyTorch 1.12.1** with **CUDA 11.3** 
- **PyTorch Geometric 2.1.0** 

### Step 1: Clone the repository 

```bash 
git clone https://github.com/WenqiZhang-HIT/MGDN-project.git 
cd MGDN-project
```

### Step 2: Install dependencies using `install_packages.py`

The project provides a script to install the required dependencies automatically. Run the following command to install the necessary libraries:

```bash
python install_packages.py
```

## Step 3: Install your datasets

Place your dataset in the correct path under the `./data/your_datasets_name` folder. For datasets requiring preprocessing, tools are provided in the `./util` folder.

You can apply for the publicly available SWaT dataset by visiting the following link:

https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

## Step 4: Run the model

```bash
python main.py
```

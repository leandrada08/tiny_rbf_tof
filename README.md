# **Self-Calibrating ToF Sensor by means of On-Device Learning**

This repository contains the source code for the **Self-Calibrating ToF Sensor by means of On-Device Learning** project. In this work, we explore the viability of applying a neural network method on a ToF (Time-of-Flight) sensor to improve its calibration and performance, comparing it with conventional approaches.

---

## **Table of Contents**

- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
- [Environment Setup](#environment-setup)
- [Authors](#authors)

---

## **Repository Structure**

The repository is organized as follows:

- [`data`](data/): Contains the datasets used in the project’s development and the Jupyter Notebooks for data preprocessing.  
    - [`ret_shape.txt`](data/ret_shape.txt/): Reference signal emitted by a ToF sensor, used for creating synthetic data.  
    - [`data_set_tof.ipynb`](data/data_set_tof.ipynb): Notebook for dataset generation.  
    - [`test_data_200cm.csv`](data/test_data_200cm.csv)  
    - [`train_data_200cm.csv`](data/train_data_200cm.csv)
- [`notebooks`](notebooks/): Jupyter Notebooks for running experiments.  
- [`src`](src/): Python source code for various implementations of neural networks, including `MRAN`, `GGAP`, and `OnlineRBF` (the proposed method), as well as auxiliary functions.

---

## **Experiments**

Since we are in an early phase, the main objective is to evaluate whether the neural network method is viable for calibrating the ToF sensor. To achieve this, we perform comparisons with conventional techniques and analyze its performance in different scenarios.

The implemented experiments include:

- [`benchmarks`](notebooks/benchmarks.ipynb): Evaluation of `MRAN`, `GGAP`, and `OnlineRBF` on reference functions.  
- [`network_rbf_tof`](notebooks/network_rbf_tof.ipynb):
   1. Experimentation with the `OnlineRBF` network applied to the ToF sensor.  
   2. Evaluation of the ToF sensor using traditional calibration and processing methods.  
   3. Comparison of the neural network-based method with traditional approaches in different test scenarios.

---

## **Environment Setup**

To work on this project, follow these steps to manually set up the environment:

### **Prerequisites**

- Install [Visual Studio Code](https://code.visualstudio.com/) or another IDE compatible with Python.  
- Install and configure [Python](https://www.python.org/downloads/).  
- Add the recommended extensions for working with Python and Jupyter Notebooks.

### **Installation Steps**

1. Clone the repository:

   ```bash
   git clone https://github.com/usuario/self-calib-ToF-odl.git
   ```

2. Install Python dependencies:

   ```bash
   cd py
   pip install --user -r requirements.txt
   ```

---

## **Authors**

- Luis Elain Andrada ([l.andrada@unisa.studenti.it](mailto:l.andrada@unisa.studenti.it))

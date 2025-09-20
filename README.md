# Fuzzy Systems and Computational Intelligence Projects  

This repository contains coursework projects completed at **Aristotle University of Thessaloniki** for the course *Fuzzy Systems and Computational Intelligence* of the **Electrical and Computer Engineering Department**.<br>
Each project applies fuzzy logic and computational intelligence methods to solve real-world problems in control, regression, and classification.  

---

## 📡 **Project 1 – Satellite Control (PID vs Fuzzy Controller)**  
- **Goal**: Compare a classical PI/PID controller with a Fuzzy Logic Controller (FLC) in controlling a satellite system.  
- **Approach**:  
  - Designed and tuned PI controllers using MATLAB Control Toolbox and root locus analysis.  
  - Built a fuzzy controller (FZ-PI) with appropriate membership functions and fuzzy rules.  
  - Simulated and compared performance (step responses, error dynamics, stability).  
- **Outcome**: Fuzzy control improved response handling of nonlinearities compared to classical PI/PID.  

---

## 🚗 **Project 2 – Car Control with Fuzzy Logic**  
- **Goal**: Implement a Fuzzy Logic Controller to control the position and heading of a car toward a target point.  
- **Approach**:  
  - Modeled car movement with position `(x, y)` and heading angle θ.  
  - Defined fuzzy inputs: distance in x, distance in y, and heading error.  
  - Designed fuzzy rules for velocity and steering correction.  
  - Implemented and tested using MATLAB’s FIS Editor.  
- **Outcome**: Fuzzy logic successfully guided the car to the target while handling uncertainty and nonlinear behavior.  

---

## 📊 **Project 3 – Regression with TSK Models**  
- **Goal**: Apply Takagi–Sugeno–Kang (TSK) fuzzy models to solve nonlinear regression problems.  
- **Approach**:  
  - **Phase 1**: Used the [Airfoil Self-Noise dataset](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) (UCI Repository) to train and evaluate TSK models with different output types (singleton vs polynomial) and varying numbers of membership functions.  
  - **Phase 2**: Applied TSK models to the high-dimensional [Superconductivity dataset](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data) (UCI Repository).  
    - Preprocessing: feature selection (Relief, mRMR, FMI), clustering-based rule generation, grid search, and cross-validation.  
- **Outcome**: Demonstrated trade-offs between model complexity, rule explosion, and prediction accuracy.  

---

## 🧩 **Project 4 – Classification with TSK Models**  
- **Goal**: Investigate the ability of Takagi–Sugeno–Kang (TSK) fuzzy models in solving classification problems.  
- **Approach**:  
  - **Phase 1**: Applied TSK models to the [Haberman’s Survival dataset](https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival) (UCI Repository).  
    - Split data into training, validation, and test sets (60/20/20).  
    - Trained TSK models with varying numbers of fuzzy IF-THEN rules, using *Subtractive Clustering* in both **class-independent** and **class-dependent** modes.  
    - Optimized parameters with hybrid learning (backpropagation for membership functions, least squares for consequents).  
    - Evaluated performance with error/confusion matrices, overall accuracy (OA), producer’s and user’s accuracy, and Cohen’s kappa (κ̂).  
  - **Phase 2**: Applied TSK models to the high-dimensional [Epileptic Seizure Recognition dataset](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition) (UCI Repository).  
    - Addressed rule explosion via **feature selection** (Relief, mRMR, FMI) and clustering-based rule generation.  
    - Performed **grid search** combined with **5-fold cross-validation** to tune parameters (number of selected features, cluster radius).  
    - Compared performance across parameter choices and visualized learning curves and fuzzy set evolution.  
- **Outcome**: Showed that TSK models can be extended beyond regression into classification. Trade-offs were observed between rule interpretability, feature reduction, and classification accuracy.  

---

## 🛠️ **Technologies Used**  
- MATLAB (Control Toolbox, Fuzzy Logic Toolbox)  
- UCI Datasets ([Airfoil Self-Noise](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise), [Superconductivity](https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data), [Haberman’s Survival](https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival), [Epileptic Seizure Recognition](https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition))  

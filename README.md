# eRBCDet
erythrocyte RNA-based cancer detection model

## Requirements

The main requirements are listed below:

* Python 3.7
* Numpy
* Keras 2.0.4
* Scikit-learn 0.21
* tensorflow 1.5.0

## The description of CRCCP source codes

* DNN.py

    The code is used for Deep Neural Networks (DNN) model training and cancer-specific model integration.

* SVM.py

    The code is used for Support Vector Machine (SVM) model training.
  
* RF.py

    The code is used for Random Forest (RF) model training.

* GNB.py

    The code is used for Gaussian Naive Bayes (GNB) model training.

* PLR.py

    The code is used for Penalized Logistic Regression (PLR) model training and pan-cancer model integration.

* ROC.py

    The code is used to illustrate the receiver operating characteristic (ROC) curve based on sensitivity and 1-specificity scores, and compute the AUC value.

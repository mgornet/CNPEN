# CNPEN

During my work at the CNPEN - the French "National Pilot Committee for Digital Ethics" - I was asked to look into facial recognition technologies and study them from an ethical standpoint.

I thus developped a simple program to train and test a facial recognition algorithm, then hightlight the ethical issues that arise.

## 1. Train a model

Clone the repository on your own laptop or export it on a remote server (Google Colab, Kaggle Kernel etc).

Launch the training via the notebook "train". You can download a file of your model parameters at the end.

## 2. Determine the best threshold for your model

Run the notebook "determine_threshold" to visualize the different thresholds that can be used for your model. You can upload your own model at the beginning of the notebook or use a pretrained model.

## 3. Test your model

Run the notebook "various_tests" and observe the results. You can make your own tests depending on what you want to study. You can upload your own model at the beginning of the notebook or use a pretrained model.

## 3. Check the fairness of your model

Run the notebook "fairness study" to look at some metrics showing how fair is your model. You can upload your own model at the beginning of the notebook or use a pretrained model.

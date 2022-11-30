# CNPEN

During my work at the CNPEN - the French "National Pilot Committee for Digital Ethics" - I was asked to look into facial recognition technologies and study them from an ethical standpoint.

With Claude Kirchner and Catherine Tessier, we developped a simple program to train and test a facial recognition algorithm, then hightlight the ethical issues that arise. In that end, we have choosen technical several parameters to assess their impact on efficiency and fairness.

A technical report of my time at the CNPEN can be found in this repository.

We shared our first results in a short [article](https://ercim-news.ercim.eu/en131/special/operational-fairness-for-facial-authentication-systems) of ERCIM News.

An academic article is currently being written.

## 1. Train a model

Clone the repository on your own laptop or export it on a remote server (Google Colab, Kaggle Kernel etc).

Make sure you have the correct environement.

Launch the training via the notebook "train". You can download a file of your model parameters at the end.

Alternatively, you can lauch the python script train.py

## 2. Determine the best threshold for your model

Run the notebook "determine_threshold" to visualize the different thresholds that can be used for your model. You can upload your own model at the beginning of the notebook or use a pretrained model.

## 3. Test your model

Run the notebook "various_tests" and observe the results. You can make your own tests depending on what you want to study. You can upload your own model at the beginning of the notebook or use a pretrained model.

Alternatively, you can lauch the python script test.py

## 3. Check the fairness of your model

Run the notebook "fairness study" to look at some metrics showing how fair is your model. You can upload your own model at the beginning of the notebook or use a pretrained model.

Run the "Data_analysis" notebook to visualize the different efficiency metrics on a graph.

Alternatively, you can lauch the python script fairness.py

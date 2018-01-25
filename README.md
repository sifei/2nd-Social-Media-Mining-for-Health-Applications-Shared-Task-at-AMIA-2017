# UKNLP team's supervised models for the 2nd Social Media Mining for Health Applications Shared Task at AMIA 2017

This repository contains code for task 1 and 2 of the 2nd Social Media Mining for Health Applications shared task at AMIA 2017 which employed supervised framework for identifying ADR mentions and classifying medication intake messages on Twitter. The tasks are described below with more documentation available in their respective subdirectories.

For info on the related shared task, please refer to: https://healthlanguageprocessing.org/sharedtask2/

## Task 1

The first task is binary classification of twitter posts for mentions of adverse drug reactions (ADRs). The program takes as input a post and returns a binary 0/1 indicating whether it or not contains ADR mentions.

In this task our best model is an ensemble involving both a traditional machine learning approach (logistic regression) and a deep learning approach (CNN with attention). 

The [task1](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task1) folder contains the linear and deep model submitted for the shared task. An example of input file format is also provided at [task1/data_sample](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task1/data_sample) folder.

## Task 2
The second task is classification of posts describing medication intake. Given a twitter post, the program returns one of three possible categorical labels:

- *personal medication intake* – tweets in which the user clearly expresses a personal medication intake/consumption
- *possible medication intake* – tweets that are ambiguous but suggest that the user may have taken the medication
- *non-intake* – tweets that mention medication names but do not indicate personal intake

In this task our best model is an ensemble involving 10 deep learning models (CNN with attention). 

The [task2](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task2) folder contains the linear and deep model submitted for this task. An example of input file format is also provided at [task2/data_sample](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task2/data_sample) folder.

## Acknowledgements

Please consider citing the following paper(s) if you use this software in your work:

> Sifei Han, Tung Tran, Anthony Rios, and Ramakanth Kavuluru. "Team UKNLP: Detecting ADRs, Classifying Medication Intake Messages, and Normalizing ADR Mentions on Twitter" In Proceedings of the 2nd Social Media Mining for Health Research and Applications Workshop
co-located with the American Medical Informatics Association Annual Symposium (AMIA 2017), vol-1996, pp. 49-53. 2017.

```
@inproceedings{han2017team,
  title={Team UKNLP: Detecting ADRs, Classifying Medication Intake Messages, and Normalizing ADR Mentions on Twitter},
  author={Han, Sifei and Tran, Tung and Rios, Anthony and Kavuluru, Ramakanth},
  booktitle={Social Media Mining for Health Research and Applications},
  pages={49--53},
  year={2017},
  organization={AMIA}
}
```

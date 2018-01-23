# Deep model for 2nd Social Media Mining for Health Applications Shared Task at AMIA 2017

## Task1
Automatic classification of adverse drug reaction (ADR) mentioning posts. It is a binary classification.

In this task our best model is ensemble the traditional machine learning approach (logisitic regression) with deep learning model (CNN with attention). 

The [task1](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task1) folder contains the linear model and deep model we used for the shared task. The example of input file fomrat also provided at [data_sample](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task1/data_sample) folder.

## Task2
Automatic classification of posts describing medication intake. It is a three-class classification.

In this task our best model is averaging 10 deep learning models (CNN with attention). 

The [task2](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task2) folder contains the linear model and deep model we used for the shared task. The example of input file fomrat also provided at [data_sample](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task2/data_sample) folder.


More shared task description please ref: https://healthlanguageprocessing.org/sharedtask2/


## Acknowledgements

Please consider citing the following paper(s) if you use this software in your work:

> Sifei Han, Tung Tran, Anthony Rios, and Ramakanth Kavuluru. "Team UKNLP: Detecting ADRs, Classifying Medication Intake Messages, and Normalizing ADR Mentions on Twitter" In Proceedings of the 2nd Social Media Mining for Health Research and Applications Workshop
co-located with the American Medical Informatics Association Annual Symposium (AMIA 2017), vol-1996, pp. 49-53. 2017.

```
@inproceedings{kavuluru2017extracting,
  title={Team UKNLP: Detecting ADRs, Classifying Medication Intake Messages, and Normalizing ADR Mentions on Twitter},
  author={Han, Sifei and Tran, Tung and Rios, Anthony and Kavuluru, Ramakanth},
  booktitle={Social Media Mining for Health Research and Applications},
  pages={49--53},
  year={2017},
  organization={AMIA}
}
```


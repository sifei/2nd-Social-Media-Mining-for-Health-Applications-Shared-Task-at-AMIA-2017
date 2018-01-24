# Deep model for 2nd Social Media Mining for Health Applications Shared Task at AMIA 2017

Implementation of traditional machine learning method (logisitic regression).
 * By default, training, deveploment and test set on ``data_sample`` directory.

## Required Packages
- Python 2.7
- numpy
- gensim
- sklearn
- nltk

## Usage

Please note that this model takes as input the path of the data folder and not paths to each individual file. A data folder is expected to have the following files: `task1_full_text.txt`, `task1_full_label.txt`, and `task1_test_text.txt`. Please see the example_dataset directory for an example of the format. 

- More info about the dataset format can be found [here](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task1/data_sample).



### Training and Evaluating
Create feature set
```
python build_feature.py
```
Get prediction probability
```
python predict.py
```


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


You can get Twitter Word2vec model here:
https://www.fredericgodin.com/software/https://nlp.stanford.edu/projects/glove/

## Author

> Sifei Han
> sehan2 **[at]** g.uky.edu




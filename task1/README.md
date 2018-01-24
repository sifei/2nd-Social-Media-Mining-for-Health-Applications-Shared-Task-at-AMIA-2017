For the task1, the best model is averaging the prediction probability between the deep model (CNN with Attention) and linear model (logistic regression).

In the TML(traditional machine learning) folder, we included the [feature extration](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/blob/master/task1/Linear_model/build_feature.py), train and get prediction probaility at [predict.py](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/blob/master/task1/Linear_model/predict.py)

In the Deep Model folder, we included the [training](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/blob/master/task1/Deep%20Model/train.py) and [testing](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/blob/master/task1/Deep%20Model/test.py) code.

More details please to go the folder.

To average the prediction probability
```
python average.py
```

The [data_sample folder](https://github.com/sifei/2nd-Social-Media-Mining-for-Health-Applications-Shared-Task-at-AMIA-2017/tree/master/task1/data_sample) contains sample inputs and output prediction result file.

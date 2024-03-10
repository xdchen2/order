# Modelling the redundancy within word order

Codes for the paper:  https://arxiv.org/abs/2402.18838

## Dataset

The dataset used in for the paper are available at https://gluebenchmark.com/ and https://super.gluebenchmark.com/.

## Generate scrambled texts

Hyperparameters can be found at ./corpus-gen/para.py

The code below will generate scrambled texts for each dataset in GLUE and SuperGLUE.

```
python ./corpus-gen/main.py

```

## Train the re-ordering model

The re-ordering model takes in scrambled texts and output the orginal texts. The model structure is T5.

```
python ./probe/main.py

```

## Calculate the mutual information with the re-ordering model

The mutual information between scrambled texts and orginal texts is estimated with a re-ordering model and a pre-trained language model. The re-ordering model is trained with the code in *probe*, while the language model is available at huggingface.

```
python ./mi/main.py

```


## Calculate accuracy for GLUE and SuperGLUE on scrambled texts

For each scrambled dataset, the code below will generate the corresponding accuracy for each task using a T5 model.

```
python ./acc/main.py

```

## Generalized mixed-effect models

The GLM model is trained using the lme4 package. The visualization is achieved with the ggplot2 package.

```
R ./R/main.R

```


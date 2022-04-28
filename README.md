<!-- # please respect the structure below-->
*See the [MediaEval 2021 webpage](https://multimediaeval.github.io/editions/2021/) for information on how to register and participate.* <br>
*See the [Sport Task MediaEval 2021 webpage](https://multimediaeval.github.io/editions/2021/tasks/sportsvideo/) for information on the task.*

## Introduction
This task offers researchers an opportunity to test their fine-grained classification methods for detecting and recognizing strokes in table tennis videos. (The low inter-class variability makes the task more difficult than with usual general datasets like UCF-101.) The task offers two subtasks:

***Subtask 1: Stroke Detection:*** Participants are required to build a system that detects whether a stroke has been performed, whatever its class, and to extract its temporal boundaries. The aim is to be able to distinguish between moments of interest in a game (players performing strokes) from irrelevant moments (between strokes, picking up the ball, having a break…). This subtask can be a preliminary step for later recognizing a stroke that has been performed. 

***Subtask 2: Stroke Classification:*** Participants are required to build a classification system that automatically labels video segments according to a performed stroke. There are 20 possible stroke classes. 

Compared with [Sports Video 2020](https://multimediaeval.github.io/editions/2020/tasks/sportsvideo/), this year we extend the task in the direction of detection and also enrich the dataset with new and more diverse stroke samples. The overviw paper of the task is already available [here](https://www.labri.fr/projet/AIV/MediaEval/Sports_Video_Task_2021.pdf).

Participants are encouraged to make their code public with their submission.

## Baseline
In order to help participants in their submission, to process videos, annotation files and deep learning techniques, we provide a baseline in this git which is formated to process the provided data by the task organizers.

The method is simple and is based on the following submission using only RGB data:

Pierre-Etienne Martin, Jenny Benois-Pineau, Boris Mansencal, Renaud Péteri, Julien Morlier. Siamese Spatio-temporal convolutional neural network for stroke classification in Table Tennis games. MediaEval 2019 Workshop, Oct 2019, Sophia Antipolis, France. [⟨hal-02937668⟩](https://hal.archives-ouvertes.fr/hal-02937668) - [Paper here](https://hal.archives-ouvertes.fr/hal-02937668/document)

The data processing is trivial. The rgb frames are resized to [120,120] and stacked together to form tensors of length 98 following the annotation boundaries (based on begin temporal boundary) and are fed to the network:

![](RGB-STCNN-Model.png)

The training method uses nesterov momentum over a fixed amount of epoch.

The classification and detection task are dealt with similarly: we consider 20 classes for the classification task and 2 classes for the detection task. Negative samples are extracted for the detection class and negative proposals are build on the test set.
Training method is similar too.

This repository is not meant to lead to good performance but to provide a skeleton of a method to help the participants. The workflow is based on OpenCV and PyTorch. The requierment are provided in the [requierments.txt file](requirements.txt). Use:

``` bash
python3 main.py
```

main.py shall be adapted according to your tree. We encourage participants to fork this repository and share their contributions. Best method (if easily reproducible) may be used as a baseline for next year.

Thank you for your participation.

## Export env
<!-- ```
conda env export --name torch_env --from-history --file environment.yml
``` -->

```
conda env create --prefix ./env --file environment.yml
conda activate ./env
```



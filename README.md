<!-- # please respect the structure below-->
*See the [MediaEval 2022 webpage](https://multimediaeval.github.io/editions/2022/) for information on how to register and participate.* <br>
*See the [Sport Task MediaEval 2022 webpage](https://multimediaeval.github.io/editions/2022/tasks/sportsvideo/) for information on the task.*

# Introduction

Running since 2019, this task was focused during the first two years on classification of temporally segmented videos of single table tennis strokes.
Since the third edition of the task, two subtasks have been proposed. The dataset also has been enriched this year with new and more diverse stroke samples.

***Subtask 1 :*** is a classification task: participants are required to build a classification system that automatically labels video segments according to a performed stroke. There are 20 possible stroke classes and an additional non-stroke class.

***Subtask 2 :***  is a more challenging subtask proposed since last year: the goal here is to detect if a stroke has been performed, whatever its classes, and to extract its temporal boundaries. The aim is to be able to distinguish between moments of interest in a game (players performing strokes) from irrelevant moments (picking up the ball, having a break…). This subtask can be a preliminary step for later recognizing a stroke that has been performed.
 

The organizers encourage the use of the method developed for subtask 1 to solve subtask 2. Participants are also invited to use the provided baseline as a starting point in their investigation. Finally, participants are encouraged to make their code public with their submission.

# Baseline
In order to help participants in their submission, to process videos, annotation files and deep learning techniques, we provide a baseline in this git which is formated to process the provided data by the task organizers.

The method is simple and is based on the following method using only RGB data:

Pierre-Etienne Martin, Jenny Benois-Pineau, Renaud Péteri, Julien Morlier. 3D attention mechanism for fine-grained classification of table tennis strokes using a Twin Spatio-Temporal Convolutional Neural Networks. 25th International Conference on Pattern Recognition (ICPR2020), Jan 2021, Milano, Italy. [⟨hal-02977646⟩](https://hal.archives-ouvertes.fr/hal-02977646) - [Paper here](https://hal.archives-ouvertes.fr/hal-02977646/document)

The data processing is trivial. The rgb frames are resized to a width of 640 and stacked together to form tensors of length 98 following the annotation boundaries. Data augmentation is used in order to start at different time point, perform some spatial transformation and increase variability. The tensros are fed to a simple Network using attention mechanisms. TODO: add image

![](RGB-ASTCNN-Model.png)

The training method uses nesterov momentum over a fixed amount of epoch. The learning rate is modified according to the loss evolution. The model with best performance on the validation loss is saved. Training method are similar for both subtasks.

We consider 21 classes for the classification task and 2 classes for the detection task. Negative samples are extracted for the detection class and negative proposals are build on the test set. Test is performed with the trimmed proposal (with one window centered or with a sliding window and several post processing approaches) or by running a sliding window on the whole video for the detection task. The lastest output is processed in order to segment in time strokes. Too short strokes are not considered. The model trained on the classification task is also used on the segmentation task without further training on the detection data.

This repository is not meant to lead to good performance but to provide a skeleton of a method to help the participants. The workflow is based on OpenCV and PyTorch. The requierment are provided in the [environment.yml file](environment.yml).

## Download the data
<!--
For organizers:
``` bash
# to create md5sums files
find -type f -exec md5sum "{}" + > MD5SUMS
```
-->
To have access to the data:
1- register to MediaEval workshop (free) with your **institutional** email address (gmail, gmx, outlook addresses... will not be accepted). See the [MediaEval website](https://multimediaeval.github.io/editions/2022/) to register.
2- send to the MediaEval organizers the [MediaEval data agreement](https://multimediaeval.github.io/editions/2021/docs/MediaEval2022_UsageAgreement.pdf) filled for our task. Instructions are on the document.
3- by accessing the data, you agree to the [Particular Conditions](Particular_conditions2022.pdf) of usage of the data.
To formally accept them, send an email to ![](image1.png) with: « I Accept the TTStroke-21 Particular Conditions 2022 » in the email body.

Once these 3 steps performed, you will receive an username and password to download the data.
Use the following commands to clone this repo and download the data (replace `USER` and `PASSWORD`).

``` bash
# Clone baseline repo
git clone https://github.com/ccp-eva/SportTaskME22.git
cd SportTaskME22
# Download data in repo
wget --user=USER --password='PASSWORD' -r -np -nH --cut-dirs=4 https://www.labri.fr/projet/AIV/MediaEval/2022/data/
# Check download is correct
cd data/
md5sum -c MD5SUMS
# Remove potential index files
find . -type f -name "index*" -exec rm {} \;
cd ..
```

It should create a directory "data/", with two subdirectories: "classificationTask" and "detectionTask". Another directory <Team_name> is provided meant to be filled for your submition.
In each task directory, you should have subdirectories dedicated to each set.
We advise not to modify these directories to be able to run properly the provided scripts.
Make sure to check that files were downloaded correctly using MD5SUM.
Make sure to delete the potential index files for running properly the check and evaluation scripts.

## Run the baseline
<!--
For organizers:
``` bash
# to create env .yml files
conda env export --name ME22_env --file environment_with_versions.yml
conda env export --name ME22_env --from-history --file environment.yml
```
-->

Please wisit [ANACONDA website](https://www.anaconda.com/) for its installation in order to replicate the running enviroment. 
Then install the required conda environment using the .yml files.
We also advise to use [screen](https://www.gnu.org/software/screen/manual/screen.html) to manage your terminals and be able to run long processes in the backgroud.

``` bash
# Create the conda environment (you can change the name and location)
conda env create --prefix ./env --file environment_with_versions.yml # or environment.yml
# Activate the conda environment
conda activate ./env
# Run the baseline (takes several hour, you may use screen)
python main.py
```

main.py shall be adapted according to your tree. We encourage participants to fork this repository and share their contributions. Best method (if easily reproducible) may be used as a baseline for next year.

You may having trouble working with gpu and pytorch even with the installation steps. This might be due to your platform. You can reinstall pytorch using the instruction on [the pytorch website](https://pytorch.org/). You may also check the [CUDA installation guideline](https://docs.nvidia.com/cuda/).

# Submittion

Please, insert your run in the <Team_name> folder (that you may rename with the name of your team) according to the task. The name of the runs may vary but you may use simple digits. Zip the folder and sent it to ![](image1.png).

Before sending your submission, check if the format you will send is correct by running from the data/ folder:

``` bash
python3 verif_submission.py <Team_name>
```

You may check similarly with the validation (or train) set by running:

``` bash
python3 verif_submission.py <Team_name> validation
```

You may check performances with validation (or train) set too by running:

``` bash
python3 evaluation.py <Team_name> validation
```

Similar scripts will be used to compare the test ground truth and your submission. Do not hesitate to give your feedback on the provided scripts.

The submission folder should look like:

TODO: add image tree

`<Team_name>/<task>/<run_id>/<xml_file(s)_from_test_set>`

For example:

- `Team1`
	- `classificationTask`
	   - `1`
	     - `<xml_file_from_test_set>`
	   - `2`
	     - `<xml_file_from_test_set>`
	   - `3`
	     - `<xml_file_from_test_set>`
	   - `4`
	     - `<xml_file_from_test_set>`
	   - `5`
	     - `<xml_file_from_test_set>`
	- `detectionTask`
	   - `1`
	     - `<xml_file1_from_test_set>`
	     - `<xml_file2_from_test_set>`
	     - ...
	   - `2`
	     - `<xml_file1_from_test_set>`
	     - `<xml_file2_from_test_set>`
	     - ...
	   - `3`
	     - `<xml_file1_from_test_set>`
	     - `<xml_file2_from_test_set>`
	     - ...
	   - `4`
	     - `<xml_file1_from_test_set>`
	     - `<xml_file2_from_test_set>`
	     - ...
	   - `5`
	     - `<xml_file1_from_test_set>`
	     - `<xml_file2_from_test_set>`
	     - ...

Thank you for your participation.

## To cite this work

To cite this work, we invite you to include some previous work. Find the bibTex below. TODO: Update with baseline working note and baseline papers ref.

```
@inproceedings{PeICPR:2020,
  author    = {Pierre{-}Etienne Martin and
              Jenny Benois{-}Pineau and
              Renaud P{\'{e}}teri and
              Julien Morlier},
  title     = {3D attention mechanisms in Twin Spatio-Temporal Convolutional Neural Networks. Application to  action classification in videos of table tennis games.},
  booktitle = {{ICPR}},
  publisher = {{IEEE} Computer Society},
  year      = {2021},
}

@phdthesis{PeThesis:2020,
  author    = {Pierre{-}Etienne Martin},
  title     = {Fine-Grained Action Detection and Classification from Videos with
               Spatio-Temporal Convolutional Neural Networks. Application to Table
               Tennis. (D{\'{e}}tection et classification fines d'actions {\`{a}}
               partir de vid{\'{e}}os par r{\'{e}}seaux de neurones {\`{a}}
               convolutions spatio-temporelles. Application au tennis de table)},
  school    = {University of La Rochelle, France},
  year      = {2020},
  url       = {https://tel.archives-ouvertes.fr/tel-03128769},
  timestamp = {Thu, 25 Feb 2021 08:50:23 +0100},
  biburl    = {https://dblp.org/rec/phd/hal/Martin20a.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}VERSION = {v1},
}
```


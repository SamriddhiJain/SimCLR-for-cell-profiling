# SimCLR for cell-profiling
This repo hosts the code for all the experiments done as part of the group project "SimCLR for cell-profiling" in the course Deep Learning offered at ETH Zurich Autumn Semester 2020.

## Project overview

![](fig/fig.png)

In this project we are attempting to improve single-cell representations by applying the SimCLR framework to multi-channel images of our “field-of-views”. A field-of-view is the underlying physical sample (composed of cells) of an image. The different channels represent fluorescence bound DNA, Tubulin and Actin. The original images are cropped to provide single-cell images by using cell locations as described in Ljosa et al. [2]. SimCLR is applied to single-cell images to extract appropriate single-cell representations. The representations of cells in a same field-of-view are aggregated by taking their mean morphological profile. The quality of mean profiles (i.e. mean representations) is evaluated based on NSC and NSCB metrics. Exploration and visualization of mean profiles is also part of our project.

## Datasets
The methods are evaluated on BBBC021 (Caie et al. [1]). This dataset consists of images captured from MCF-7 breast cancer cell populations exposed to a group of chemical compounds for a fixed amount of time. A subset of BBBC021 has previously been investigated and annotated for MOAs by Ljosa et al. [2]. Our project uses this particular subset that provides an MOA label as well as a set of single-cell locations for each field-of-view.

## Code & experiments
### Installation
```
$ conda env create --name simclr --file env.yml
$ conda activate simclr
```

### Running Experiments
The SimCLR training code is based on pytorch and is located in the directory `src`. All the experiments can be tuned using `config.yaml`. Follow these steps to run any desired experiments,
- Update the entries in `config.yaml` file as per the experiment. The parameters are discussed in more details in the following sections.
- Use `python run.py` to train the SimCLR framework, the trained models will be stored in `runs` directory.
- To evaluate NSC-NSCB scores, update the model path and epoch ids in `evaluation.py`. This also needs the config file in which one can specify the test dataset paths. The script will generate NSC-NSCB scores with and without TVN and the related tSNE visualizations. This script also saves the image representations in specified folder. If you just need to generate single cell embeddings, you can directly use `generate_features.py` with the config file.
- *Example Evaluation:* 
  - We provide a small set of [validation data](https://polybox.ethz.ch/index.php/s/xq7uhAwkZAu2UQR) to run a quick evaluation. Download the data and update the path in `eval_dataset` section in `config.yaml`. 
  - Place the model checkpoints for creating representations in `runs\{model_name}\checkpoints\` directory, they are expected to be in format `model_epouch_{epoch_number}.pth`. Update the `{model_name}` and `{epoch_number}`(s) in the file `evaluation.py`. 
  - We also provide [pre-trained weights](https://polybox.ethz.ch/index.php/s/y2EJknOtLL6B8EF) for our best model to reproduce the results.

```
$ cd src
$ python run.py         # Training SimCLR
$ python evaluation.py  # NSC-NSCB evaluations
```

### Repository Structure
- All the SimCLR training and evaluation code is in the directory `src`. The `archive` folder has some preliminary code which we started with but not used for final evaluations. The next points explain the implementation of SimCLR modules within the `src` folder.
- The `data_aug` module holds the code for reading the single cell image dataset, loading it as dataloaders and applying various augmentations.
- `loss` module holds the implementation of NTXent loss and LARS optimizer.
- The module `models` holds implementation of base networks (4 layer CNN) and resnet models.
- `feature_eval\random_forest_classifier.py` holds the classfier code that can be used on top of the representations to test progress. Currently we are using a random forest classifier, but a simple linear head can also be used.
- Finally, `simclr.py` binds all these modules together and holds the code for overall training.
- The trained model checkpoints are stored in directory `{working_directory}\runs`.
- The code for running downstream task (single cell representation aggregation and KNN training) is present in `evaluation.py`. This script is a wrapper over `generate_features.py` and `feature_eval\profile_measures.py` which generate single cell embeddings and run nsc evaluation pipeline respectively. Both of the scripts can also be used as stand alone modules for intermediate results. `profile_measures.py` also contains tsne visualization code ran on the aggregated data.

### Config hyperparameters
The config file and various hyperparameters are explained details below.

```yaml
# A batch size of N, produces 2 * (N-1) negative samples. Original implementation uses a batch size of 8192
batch_size: 512

# Number of epochs to train
epochs: 400

# Frequency to evaluate the similarity score using the validation set
eval_every_n_epochs: 1

# Specify a folder containing a pre-trained model to fine-tune. If training from scratch, pass None.
fine_tune_from: 'resnet-18_80-epochs'

# Frequency to which tensorboard is updated
log_every_n_steps: 50

# l2 Weight decay magnitude, original implementation uses 10e-6
weight_decay: 10e-6

# if True, training is done using mixed precision. Apex needs to be installed in this case.
fp16_precision: False

# Frequency to evaluate classifier
eval_classifier_n_epoch: 50

# Frequency to store checkpoints
checkpoint_every_n_epochs: 50

# Model related parameters
model:
  # Output dimensionality of the embedding vector z. Original implementation uses 2048
  out_dim: 256

  # The ConvNet base model. Choose one of: "resnet18" or "resnet50" or "resnet101". Original implementation uses resnet50
  base_model: "resnet50"

# Dataset related parameters
dataset:
  # path for csv metadata file
  path: '../single-cell-sample-train/sc-metadata.csv'

  # path where cell images are stored
  root_dir: '../single-cell-sample-train/'

  # jitter parameter
  s: 1

  # dataset input shape. For datasets containing images of different size, this defines the final
  input_shape: (96,96,3)

  # Number of workers for the data loader
  num_workers: 0

  # Size of the validation set in percentage
  valid_size: 0.05

  # data sampler: weighted or random,
  # weighted sampler guaranties to create balanced batches and can be used for training with full DMSO datasets
  sampler: "random"

  # whether to preload the images before training or on the fly
  preload: True

# dataset for training the classfier
# all the parameters within this field are same as the dataset field
# this field will also be used for generating embeddings in generate_features.py
eval_dataset:
  path: '../single-cell-sample-train/sc-metadata.csv'
  root_dir: '../single-cell-sample-train/'
  input_shape: (96,96,3)
  num_workers: 0
  valid_size: 0.3
  sampler: "random"
  preload: True

# NTXent loss related parameters
loss:
  # Temperature parameter for the contrastive objective
  temperature: 0.5

  # Distance metric for contrastive loss. If False, uses dot product. Original implementation uses cosine similarity.
  use_cosine_similarity: True
```

## External Code References
- We use the SimCLR pytorch code available at [SimCLR training in pytorch](https://github.com/sthalles/SimCLR) as starter code and build on top of it to run experiments specifically for our dataset. We also referred to this [repo](https://github.com/Spijkervet/SimCLR) for SimCLR training tricks, but eventually didn't use their code.
- For generating single cell images we use the code provided at [Deep Profiler](https://github.com/cytomining/DeepProfiler) from the baseline [2].
- We use the nsc and nscb evaluation code from [Deep profiler experiments](https://github.com/broadinstitute/DeepProfilerExperiments).

## References
[1] Caie, P.D. et al. (2010) High-Content Phenotypic Profiling of Drug Response Signatures across Distinct Cancer Cells. Molecular Cancer Therapeutics, 9, 1913–1926.

[2] Ljosa, V. et al. (2013) Comparison of Methods for Image-Based Profiling of Cellular Morphological Responses to Small- Molecule Treatment. J Biomol Screen. 18(10):1321–9. doi: 10.1177/1087057113503553.

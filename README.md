# Multi-Layer Materials Science Corpus - Experiment Resources

This repository contains the companion material for the following publication:

> Timo Pierre Schrader, Matteo Finco, Stefan Grünewald, Felix Hildebrand, Annemarie Friedrich. **MuLMS: A Multi-Layer Annotated Text Corpus for Information Extraction in the Materials Science Domain** WIESP 2023.

Please cite this paper if using the dataset or the code, and direct any questions regarding the dataset
to [Annemarie Friedrich](mailto:annemarie.friedrich@uni-a.de), and any questions regarding the code to
[Timo Schrader](mailto:timo.schrader@de.bosch.com).

## Purpose of this Software

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## The Multi-Layer Materials Science Corpus (MuLMS)

The Multi-Layer Materials Science corpus (MuLMS) consists of 50 documents (licensed CC BY) from the materials science domain, spanning across the following 7 subareas: "Electrolysis", "Graphene", "Polymer Electrolyte Fuel Cell (PEMFC)", "Solid Oxide Fuel Cell (SOFC)", "Polymers", "Semiconductors" and "Steel".

For a detailed description, please refer to our [HuggingFace Dataset](https://huggingface.co/datasets/timo-pierre-schrader/MuLMS) and our [paper](https://arxiv.org/abs/2310.15569).

**NOTE: This code requires Python 3.9 oder newer. It does **not** support Python 3.8 and below.**

## Available Experiments

### Named Entity Recognition

Named entitiy (NE) recognition is a token-level tagging task and deals with tagging named entities. For instance, "WO3" is an example of a "Material" in our dataset. Because named entities occur at tokel level and can span across multiple input tokens in sentence, we model this task using two different approaches, BILOU tagging scheme + [CRF](https://en.wikipedia.org/wiki/Conditional_random_field) classification layer and dependency parsing where we treat NEs as dependencies between first and last token.

Furthermore, we provide datasets for multi-task experiments where we incorporate another related datasets and their named entities to support the classifiers in learning our NEs.

The following named entities are modeled in our dataset: **MAT, NUM, VALUE, UNIT, PROPERTY, CITE, TECHNIQUE, RANGE, INSTRUMENT, SAMPLE, FORM, DEV, MEASUREMENT**

### Relation Extraction

MuLMS provides relations between pairs of entities. There are two types of relations: measurement-related relations and further relations. The first type always starts at Measurement trigger spans, the scond type does not start at a specific Measurement annotation.

There are the following relation types in MuLMS: _hasForm_, _measuresProperty_, _usedAs_, _propertyValue_, _conditionProperty_, _conditionSample_, _conditionPropertyValue_, _usesTechnique_, _measuresPropertyValue_, _usedTogether_, _conditionEnv_, _usedIn_, _conditionInstrument_, _takenFrom_, _dopedBy_

### Measurement Classification

This task is about classifying experiment-describing sentences as _qualitative_ or _quantitative_. Whereas a _quantitative_ sentence describes technical details about measurement procedures and experiments, a _quantitative_ sentence does only describe it on a high-level, leaving out important details.

We model this task on a sentence-level basis as a ternary classification task using the tagset **MEASUREMENT, QUAL_MEAS** and **NONE**.

### Argumentative Zoning
For the argumentative zoning (AZ) part MuLMS that is presented in the related publication [MuLMS-AZ: An Argumentative Zoning Dataset for the Materials Science Domain](https://aclanthology.org/2023.codi-1.1/), please refer to the separate repository on [Github](https://github.com/boschresearch/mulms-az-codi2023), which is a submodule of this repository and hence does not need an extra download.

## Data Format

### Split Setup

Our dataset is divided into several splits, please look them up in the paper for further explanation:

* train
* tune1/tune2/tune3/tune4/tune5
* dev
* test

## Setup

Please install all dependencies or the environment as listed in [environment.yml](environment.yml) and make sure to have **Python 3.9** installed (we recommend 3.9.11). You might also add the root folder of this project to the `$PYTHONPATH` environment variable. This enables all scripts to automatically find the imports.

**Important:** Also clone the Git submodule in this repo that points to the [MuLMS-AZ Repo](https://github.com/boschresearch/mulms-az-codi2023). The code files there are required to run the experiments in this repo.

**NOTE: This code really requires Python 3.9. It does **not** support Python 3.8 and below or 3.10 and above due to type hinting and package dependencies.**

## Code

We provide bash scripts in [scripts](scripts) for each NLP task separately. Furthermore, for subtaks (e.g., multi-tasking), there are additional scripts that contain all necessary parameters. Use these scripts to reproduce the results from our paper and adapt those if you want to do additional experiments. Moreover, you can check all available settings in each Python file via `python <script_name.py> --help`.

### Transformer-based Models

We use BERT-based language models, namely [BERT](https://huggingface.co/bert-base-uncased), [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased) and [MatSciBERT](https://huggingface.co/m3rg-iitd/matscibert), as contextualized transformer LMs as basis of all our models. Moreover, we implement task-specific output layers on top of the LM. All Pytorch models can be found in every `models` subdirectory of each task.

### Multi-task Datasets

* Download the [SOFC corpus](https://github.com/boschresearch/sofc-exp_textmining_resources/tree/master/sofc-exp-corpus) and place the contents in [data/](data/).
* For [MSPT](https://github.com/olivettigroup/annotated-materials-syntheses), you need to first convert the corpus to UIMA CAS format using for example [INCEpTION](https://inception-project.github.io/). You can find all UIMA CAS span types in [source/data_handling/mspt_dataset.py](source/data_handling/mspt_dataset.py).

### Evaluation

Use the `aggregate_cv_score.py` scripts in the `evaluation` subdirectory of each task to evaluate the performance of trained models across all five folds.

## License

This software is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.
The MuLMS-AZ corpus is released under the CC BY-SA 4.0 license. See the [LICENSE](data/mulms_corpus/LICENSE) file for details.
For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).

## Citation

If you use our software or dataset in your scientific work, please cite our paper:

```
@misc{schrader2023mulms,
      title={MuLMS: A Multi-Layer Annotated Text Corpus for Information Extraction in the Materials Science Domain},
      author={Timo Pierre Schrader and Matteo Finco and Stefan Grünewald and Felix Hildebrand and Annemarie Friedrich},
      year={2023},
      eprint={2310.15569},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

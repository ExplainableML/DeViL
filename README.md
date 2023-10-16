# DeViL: Decoding Vision features into Language

[[Paper]](https://arxiv.org/abs/2309.01617)

This is the official repository for our **GCPR 2023 Oral** paper on Decoding Vision features into Language (DeViL).

![DeViL Teaser](./assets/DeViL_teaser.png)

## Getting started
To ensure you have the right environment to work on please use the file environment.yml using this command

```
conda env create -n <ENVNAME> --file environment.yml
```

## Training DeViL on image-text pairs

Datasets are implemented in the `src/datasets.py` file which contains the code for data loading and data collation.

A paired image-text dataset for training returns a dictionay with items: `{'id': image id, 'image': image, 'text': captions}`.

In the `src/datasets.py` file you can find implementations for the CC3M and MILANNOTATIONS datasets.

Once the dataset is prepared, you can run training with the `src/main.py` file. Don't forget to set the `data_root` and `logdir` arguments.

For a detailed explanation of all arguments, see: `src/main.py`.


Example command:
```
python src/main.py --data_root ./data --dataset cc3m --logdir ./results --language_model facebook/opt-125m \
--vision_backbone timm_resnet50 --token_dropout 0.5 --feature_dropout 0.5 --vision_feat_layer -1 -2 -3 -4
```

## Evaluation (NLP metrics)
To evaluate a trained model run:

```
python src/main.py
--do_eval_nlp \
--model_ckpt <path of the saved checkpoint parent folder> \
--data_root <path to dataset>
```

If you wish to evaluate descriptions of a specific layer, set the `layer` argument.

## Generation of textual descriptions

To generate textual descriptions for different layers and feature locations run:

```
python src/main.py
--do_eval_qualitative \
--model_ckpt <path of the saved checkpoint parent folder> \
--data_root <path to dataset> \
--loc_ids "-1: [[-1, -1]]"
```

For more details, see the `loc_ids`, `pool_locs`, `kernel_size` arguments in `src/main.py`.

## Generation of open-vocabulary saliency maps

To generate open-vocabulary saliency maps see [viz_notebook.ipynb](./src/viz_notebook.ipynb).

## CC3M Dataset

To obtain CC3M in webdataset format, you can use [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md).

## MILAN

To train on the [MILANNOTATIONS](https://github.com/evandez/neuron-descriptions) dataset, follow the instructions to download the dataset and change the `dataset` argument to one of the MILANNOTATION [keys](./src/milan_keys.py) (e.g. "imagenet").

You can also activate the `by_unit` argument so that this dataset is processed by neuron (aka unit) instead of by image.

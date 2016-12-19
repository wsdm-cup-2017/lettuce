# wsdm-triple-scoring

This repository contains the code for our submission of [triple scoring task](http://www.wsdm-cup-2017.org/triple-scoring.html) in [WSDM Cup 2017](http://www.wsdm-cup-2017.org/index.html).

We address the task by combining multiple neural network classifiers using gradient boosted regression trees.
Similar to [past work](http://ad-publications.informatik.uni-freiburg.de/SIGIR_triplescores_BBH_2015.pdf), we train these classifiers using the instances having single class (i.e., profession and nationality) and use them to predict the classes of instances with multiple classes.

## Installing required packages

First, the required Python packages need to be installed.

```
% pip install -r requirements.txt
```

## Building databases

The following three databases are required to train our model:

* *entity_db* stores a Wikipedia redirection structure and basic statistics
* *page_db* contains paragraphs and links in the target Wikipedia pages
* *sentence_db* stores the parsed sentences contained in the [wiki-sentences](http://broccoli.cs.uni-freiburg.de/wsdm-cup-2017/wiki-sentences) file

You also need to download a Wikipedia dump file from [Wikimedia Downloads](https://dumps.wikimedia.org/).
In our experiments, we used the Wikipedia dump generated in June 2016.

```bash
% python cli.py build_entity_db WIKIPEDIA_DUMP_FILE entity_db
% python cli.py build_page_db --category=pro WIKIPEDIA_DUMP_FILE page_db_pro.db
% python cli.py build_page_db --category=nat WIKIPEDIA_DUMP_FILE page_db_nat.db
% python cli.py build_sentence_db dataset/wiki-sentences sentence.db
```

## Training classifier using Wikipedia pages

We combine the outputs of multiple supervised classifiers to compute features for our scoring model.

The first classifier is trained using the bag-of-words (BoW) and bag-of-entities (BoE) in the target Wikipedia pages.

We train eight classifiers for each category (i.e., profession and nationality) with various training configurations.
The classifiers can be built by using the following commands:

*Preparing required data*:

```bash
% python cli.py page_classifier build_dataset --category=pro --test-size=0 page_db_pro.db page_classifier_dataset_pro_full.joblib
% python cli.py page_classifier build_dataset --category=nat --test-size=0 page_db_nat.db page_classifier_dataset_nat_full.joblib
```

*Training classifiers for profession*:

```bash
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_attention_300_full --dim-size=300
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_300_full --dim-size=300 --no-attention
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_attention_300_balanced_full --dim-size=300 --balanced-weight
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_300_balanced_full --dim-size=300 --balanced-weight --no-attention
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_entity_attention_300_full --dim-size=300 --entity-only
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_entity_300_full --dim-size=300 --entity-only --no-attention
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_entity_attention_300_balanced_full --dim-size=300 --entity-only --balanced-weight
% python cli.py train_classifier page_classifier_dataset_pro_full.joblib page_classifier_model_pro_entity_300_balanced_full --dim-size=300 --entity-only --balanced-weight --no-attention
```

*Training classifiers for nationality*:

```bash
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_attention_300_full --dim-size=300
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_300_full --dim-size=300 --no-attention
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_attention_300_balanced_full --dim-size=300 --balanced-weight
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_300_balanced_full --dim-size=300 --balanced-weight --no-attention
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_entity_attention_300_full --dim-size=300 --entity-only
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_entity_300_full --dim-size=300 --entity-only --no-attention
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_entity_attention_300_balanced_full --dim-size=300 --entity-only --balanced-weight
% python cli.py train_classifier page_classifier_dataset_nat_full.joblib page_classifier_model_nat_entity_300_balanced_full --dim-size=300 --entity-only --balanced-weight --no-attention
```

## Training classifier using co-occurrence data

The second classifier is trained using the word and entity co-occurrence data of the target entities obtained from Wikipedia.
The classifier uses [wiki-sentences](http://broccoli.cs.uni-freiburg.de/wsdm-cup-2017/wiki-sentences) to compute the co-occurrence data.

You can train the classifiers using the following commands:

*Preparing required data*:

```bash
% python cli.py coocc_classifier build_coocc_matrix coocc_matrix_win5 sentence.db --word-window=5
% python cli.py coocc_classifier build_coocc_matrix coocc_matrix_win10 sentence.db --word-window=10
% python cli.py coocc_classifier build_dataset --category=pro --test-size=0 coocc_matrix_win5 coocc_classifier_dataset_win5_pro_full.joblib
% python cli.py coocc_classifier build_dataset --category=pro --test-size=0 coocc_matrix_win10 coocc_classifier_dataset_win10_pro_full.joblib
% python cli.py coocc_classifier build_dataset --category=nat --test-size=0 coocc_matrix_win5 coocc_classifier_dataset_win5_nat_full.joblib
% python cli.py coocc_classifier build_dataset --category=nat --test-size=0 coocc_matrix_win10 coocc_classifier_dataset_win10_nat_full.joblib
```

*Training classifiers for profession*:

```bash
% python cli.py train_classifier coocc_classifier_dataset_win5_pro_full.joblib coocc_classifier_model_pro_attention_win5_300_full --dim-size=300
% python cli.py train_classifier coocc_classifier_dataset_win5_pro_full.joblib coocc_classifier_model_pro_win5_300_full --dim-size=300 --no-attention
% python cli.py train_classifier coocc_classifier_dataset_win5_pro_full.joblib coocc_classifier_model_pro_attention_win5_300_balanced_full --dim-size=300 --balanced-weight
% python cli.py train_classifier coocc_classifier_dataset_win5_pro_full.joblib coocc_classifier_model_pro_win5_300_balanced_full --dim-size=300 --balanced-weight --no-attention
% python cli.py train_classifier coocc_classifier_dataset_win10_pro_full.joblib coocc_classifier_model_pro_attention_win10_300_full --dim-size=300
% python cli.py train_classifier coocc_classifier_dataset_win10_pro_full.joblib coocc_classifier_model_pro_win10_300_full --dim-size=300 --no-attention
% python cli.py train_classifier coocc_classifier_dataset_win10_pro_full.joblib coocc_classifier_model_pro_attention_win10_300_balanced_full --dim-size=300 --balanced-weight
% python cli.py train_classifier coocc_classifier_dataset_win10_pro_full.joblib coocc_classifier_model_pro_win10_300_balanced_full --dim-size=300 --no-attention --balanced-weight
```

*Training classifiers for nationality*:

```bash
% python cli.py train_classifier coocc_classifier_dataset_win5_nat_full.joblib coocc_classifier_model_nat_attention_win5_300_full --dim-size=300
% python cli.py train_classifier coocc_classifier_dataset_win5_nat_full.joblib coocc_classifier_model_nat_win5_300_full --dim-size=300 --no-attention
% python cli.py train_classifier coocc_classifier_dataset_win5_nat_full.joblib coocc_classifier_model_nat_attention_win5_300_balanced_full --dim-size=300 --balanced-weight
% python cli.py train_classifier coocc_classifier_dataset_win5_nat_full.joblib coocc_classifier_model_nat_win5_300_balanced_full --dim-size=300 --balanced-weight --no-attention
% python cli.py train_classifier coocc_classifier_dataset_win10_nat_full.joblib coocc_classifier_model_nat_attention_win10_300_full --dim-size=300
% python cli.py train_classifier coocc_classifier_dataset_win10_nat_full.joblib coocc_classifier_model_nat_win10_300_full --dim-size=300 --no-attention
% python cli.py train_classifier coocc_classifier_dataset_win10_nat_full.joblib coocc_classifier_model_nat_attention_win10_300_balanced_full --dim-size=300 --balanced-weight
% python cli.py train_classifier coocc_classifier_dataset_win10_nat_full.joblib coocc_classifier_model_nat_win10_300_balanced_full --dim-size=300 --no-attention --balanced-weight
```

## Caching classifier results

In order to enable our software to run on a VM machine (TIRA), we cache the results of the above classifiers into one file.
The cache file can be generated with the following commands:

```bash
% python cli.py cache_classifier_results --category=pro page_db_pro.db classifier_results_pro.joblib
% python cli.py cache_classifier_results --category=nat page_db_nat.db classifier_results_nat.joblib

```

## Training scorer

We adopt the gradient boosted regression trees (GBRT) to map the outputs of the above-mentioned classifiers to the final scores.
We use two models for generating the final scores: the regression model and the binary classification model.
The regression model directly estimates the final scores (ranging from 0 to 7), whereas the classification model outputs 5 and 2 for the true and false cases, respectively.

Because the training dataset is very small, we adopt the forward feature selection to select the small set of most useful features.
The feature selection can be run using the following commands:

*Profession*:

```bash
% python cli.py scorer select_features --k-features=50 --learning-rate=0.04 --max-depth=4 -o features_pro_reg_rate0.04_depth4.json scorer_dataset_pro_reg.joblib
% python cli.py scorer select_features --k-features=50 --learning-rate=0.01 --max-depth=2 -o features_pro_bin_rate0.01_depth2.json scorer_dataset_pro_bin.joblib
```

*Nationality*:

```bash
% python cli.py scorer select_features --k-features=50 --learning-rate=0.03 --max-depth=2 -o features_nat_reg_rate0.03_depth2.json scorer_dataset_nat_reg.joblib
% python cli.py scorer select_features --k-features=50 --learning-rate=0.01 --max-depth=3 -o features_nat_bin_rate0.01_depth3.json scorer_dataset_nat_bin.joblib
```

Then, the final GBRT model can be constructed using the following commands:

*Profession*:

```bash
% python cli.py scorer train_model -f features_pro_reg_rate0.04_depth4.json --learning-rate=0.05 --max-depth=4 --min-samples-split=82 --max-features=9 --subsample=1.0 --n-estimators=3000 scorer_dataset_pro_reg.joblib scorer_model_pro_reg.pickle
% python cli.py scorer train_model -f features_pro_bin_rate0.01_depth2.json --learning-rate=0.01 --max-depth=2 --min-samples-split=22 --max-features=17 --subsample=1.0 --n-estimators=1000 scorer_dataset_pro_bin.joblib scorer_model_pro_bin.pickle
```

*Nationality*:

```bash
% python cli.py scorer train_model -f features_nat_reg_rate0.03_depth2.json --learning-rate=0.045 --max-depth=2 --min-samples-split=47 --max-features=15 --subsample=0.95 --n-estimators=3000 scorer_dataset_nat_reg.joblib scorer_model_nat_reg.pickle
% python cli.py scorer train_model -f features_nat_bin_rate0.01_depth3.json --learning-rate=0.01 --max-depth=3 --min-samples-split=27 --max-features=11 --subsample=1.0 --n-estimators=3000 scorer_dataset_nat_bin.joblib scorer_model_nat_bin.pickle
```

Now, the final scoring models (i.e., *scorer_model_pro_reg.pickle*, *scorer_model_pro_bin.pickle*, *scorer_model_nat_reg.pickle*, *scorer_model_nat_bin.pickle*) should appear in the current directory.

## Estimating final scores

The submission file containing final scores is generated using the *run* command:

*Predicting scores using the regression model*:

```bash
% python cli.py scorer run -i profession.test -i nationality.test -o OUTPUT_DIR
```

*Predicting scores using the binary model*:

```bash
% python cli.py scorer run --binary -i profession.test -i nationality.test -o OUTPUT_DIR
```

The final submission file should appear in *OUTPUT_DIR*.

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

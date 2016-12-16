# wsdm-triple-scoring

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

### Training scorer

We use the gradient boosted regression trees (GBRT) to map the outputs of the above-mentioned classifiers to the final scores,


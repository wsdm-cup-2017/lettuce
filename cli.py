# -*- coding: utf-8 -*-

import click
import joblib
import json
import logging
import multiprocessing
import cPickle as pickle

from utils.dataset_loader import DatasetLoader
from utils.entity_db import EntityDB
from utils.page_db import PageDB
from utils.sentence_db import SentenceDB

import classifier_model
import page_classifier
import coocc_classifier
import scorer


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path(), default='entity_db')
@click.option('--pool-size', default=10)
@click.option('--chunk-size', default=10)
def build_entity_db(dump_file, out_file, **kwargs):
    db = EntityDB.build(dump_file, **kwargs)
    db.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.argument('dataset_dir', type=click.Path(exists=True), default='dataset')
@click.argument('entity_db_file', type=click.Path(), default='entity_db')
@click.option('--category', default='pro', type=click.Choice(['pro', 'nat']))
@click.option('--pool-size', default=10)
@click.option('--chunk-size', default=10)
def build_page_db(dataset_dir, entity_db_file, **kwargs):
    dataset = DatasetLoader(dataset_dir)
    entity_db = EntityDB.load(entity_db_file)
    PageDB.build(dataset, entity_db, **kwargs)


@cli.command()
@click.argument('wiki_sentences_file', type=click.File(), default='dataset/wiki-sentences')
@click.argument('entity_db_file', type=click.Path(), default='entity_db')
@click.argument('out_file', type=click.Path(), default='sentence.db')
def build_sentence_db(wiki_sentences_file, entity_db_file, out_file):
    entity_db = EntityDB.load(entity_db_file)
    SentenceDB.build(wiki_sentences_file, entity_db, out_file)


@cli.command()
@click.argument('clf_dataset_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--batch-size', default=100)
@click.option('--epoch', default=1)
@click.option('--random-seed', default=0)
@click.option('--text-len', default=5000)
@click.option('--entity-len', default=2000)
@click.option('--hidden-units', default=2000)
@click.option('--dropout-keep-prob', default=0.5)
@click.option('--dim-size', default=100)
@click.option('--word-only', is_flag=True)
@click.option('--entity-only', is_flag=True)
@click.option('--attention/--no-attention', default=True)
@click.option('--attention-dim-size', default=10)
@click.option('--balanced-weight', is_flag=True)
@click.option('--optimizer', default='adam')
def train_classifier(clf_dataset_file, out_file, **kwargs):
    clf_dataset = joblib.load(clf_dataset_file, mmap_mode='r')

    classifier_model.train_model(clf_dataset, out_file, **kwargs)


@cli.command(name='search_hyper_params')
@click.argument('clf_dataset_file', type=click.Path(exists=True))
@click.option('--max-evals', default=100)
@click.option('--batch-size', default=100)
@click.option('--epoch', default=1)
@click.option('--random-seed', default=0)
@click.option('--text-len', default=5000)
@click.option('--entity-len', default=2000)
@click.option('--word-only', is_flag=True)
@click.option('--entity-only', is_flag=True)
@click.option('--attention/--no-attention', default=True)
@click.option('--balanced-weight', is_flag=True)
@click.option('--optimizer', default='adam')
def search_clf_hyper_params(clf_dataset_file, **kwargs):
    clf_dataset = joblib.load(clf_dataset_file, mmap_mode='r')

    classifier_model.search_hyper_params(clf_dataset, **kwargs)


@cli.group(name='page_classifier')
def page_classifier_group():
    pass


@page_classifier_group.command(name='build_dataset')
@click.argument('page_db_file', type=click.Path())
@click.argument('out_file', type=click.File(mode='w'))
@click.argument('dataset_dir', type=click.Path(exists=True), default='dataset')
@click.argument('entity_db_file', type=click.Path(), default='entity_db')
@click.option('--category', default='pro', type=click.Choice(['pro', 'nat']))
@click.option('--text-len', default=5000)
@click.option('--min-text-len', default=5)
@click.option('--entity-len', default=2000)
@click.option('--min-entity-len', default=5)
@click.option('--abstract-only', is_flag=True)
@click.option('--min-word-count', default=5)
@click.option('--min-entity-count', default=5)
@click.option('--in-links/--no-in-links', default=True)
@click.option('--out-links/--no-out-links', default=True)
@click.option('--dev-size', default=0.05)
@click.option('--test-size', default=0.05)
@click.option('--random-seed', default=0)
def build_page_classifier_dataset(out_file, dataset_dir, page_db_file, entity_db_file, **kwargs):
    dataset = DatasetLoader(dataset_dir)
    entity_db = EntityDB.load(entity_db_file)
    page_db = PageDB(page_db_file, 'r')

    ret = page_classifier.build_dataset(dataset, page_db, entity_db, **kwargs)
    joblib.dump(ret, out_file)


@cli.group(name='coocc_classifier')
def coocc_classifier_group():
    pass


@coocc_classifier_group.command(name='build_coocc_matrix')
@click.argument('out_file', type=click.Path())
@click.argument('sentence_db_file', type=click.Path(), default='sentence.db')
@click.option('--word-window', default=None, type=int)
@click.option('--min-word-count', default=5)
def build_coocc_matrix(sentence_db_file, **kwargs):
    sentence_db = SentenceDB(sentence_db_file, 'r')
    coocc_classifier.build_coocc_matrix(sentence_db, **kwargs)


@coocc_classifier_group.command(name='build_dataset')
@click.argument('coocc_matrix_file', type=click.Path())
@click.argument('out_file', type=click.File(mode='w'))
@click.argument('dataset_dir', type=click.Path(exists=True), default='dataset')
@click.argument('entity_db_file', type=click.Path(), default='entity_db')
@click.option('--category', default='pro', type=click.Choice(['pro', 'nat']))
@click.option('--text-len', default=5000)
@click.option('--min-text-len', default=5)
@click.option('--entity-len', default=2000)
@click.option('--min-entity-len', default=5)
@click.option('--dev-size', default=0.05)
@click.option('--test-size', default=0.05)
@click.option('--random-seed', default=0)
def build_coocc_classifier_dataset(out_file, dataset_dir, entity_db_file, **kwargs):
    dataset = DatasetLoader(dataset_dir)
    entity_db = EntityDB.load(entity_db_file)

    ret = coocc_classifier.build_dataset(dataset, entity_db, **kwargs)
    joblib.dump(ret, out_file)


@cli.group(name='scorer')
def scorer_group():
    pass


@scorer_group.command()
@click.argument('page_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.argument('dataset_dir', type=click.Path(exists=True), default='dataset')
@click.argument('entity_db_file', type=click.Path(), default='entity_db')
@click.option('--category', default='pro', type=click.Choice(['pro', 'nat']))
@click.option('-i', '--init', type=click.File())
@click.option('--pool-size', default=10)
def cache_classifier_results(page_db_file, out_file, dataset_dir, entity_db_file,
                             init, **kwargs):
    page_db = PageDB(page_db_file, 'r')
    dataset = DatasetLoader(dataset_dir)
    entity_db = EntityDB.load(entity_db_file)
    if init:
        initial_data = joblib.load(init)
    else:
        initial_data = {}

    scorer.cache_classifier_results(dataset, initial_data, out_file, page_db, entity_db, **kwargs)


@scorer_group.command(name='train_model')
@click.argument('scorer_dataset_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'))
@click.option('-f', '--feature-file', type=click.File())
@click.option('--n-estimators', default=1000)
@click.option('--learning-rate', default=0.01)
@click.option('--max-depth', default=3)
@click.option('--min-samples-split', default=30)
@click.option('--max-features', default=None, type=int)
@click.option('--subsample', default=1.0)
def train_scorer_model(scorer_dataset_file, out_file, feature_file, **kwargs):
    scorer_dataset = joblib.load(scorer_dataset_file)
    if feature_file:
        feature_list = json.load(feature_file)['features']
    else:
        feature_list = None

    ret = scorer.train_model(scorer_dataset, feature_list, **kwargs)
    pickle.dump(ret, out_file)


@scorer_group.command()
@click.argument('scorer_dataset_file', type=click.Path(exists=True))
@click.option('-f', '--feature-file', type=click.File())
@click.option('--cv', default=10)
@click.option('--n-estimators', default=1000)
@click.option('--learning-rate', default=0.01)
@click.option('--max-depth', default=3)
@click.option('--min-samples-split', default=30)
@click.option('--max-features', default=None, type=int)
@click.option('--subsample', default=1.0)
def evaluate(scorer_dataset_file, feature_file, **kwargs):
    scorer_dataset = joblib.load(scorer_dataset_file)
    if feature_file:
        feature_list = json.load(feature_file)['features']
    else:
        feature_list = None

    scorer.evaluate(scorer_dataset, feature_list, **kwargs)


@scorer_group.command(name='build_dataset')
@click.argument('clf_cache_file', type=click.Path())
@click.argument('out_file', type=click.File(mode='w'))
@click.argument('dataset_dir', type=click.Path(exists=True), default='dataset')
@click.argument('entity_db_file', type=click.Path(), default='entity_db')
@click.option('--category', default='pro', type=click.Choice(['pro', 'nat']))
@click.option('--binary/--regression', default=True)
def build_scorer_dataset(clf_cache_file, out_file, dataset_dir,
                         entity_db_file, **kwargs):
    clf_cache = joblib.load(clf_cache_file, mmap_mode='r')
    dataset = DatasetLoader(dataset_dir)
    entity_db = EntityDB.load(entity_db_file)

    ret = scorer.build_dataset(dataset, clf_cache, entity_db, **kwargs)

    joblib.dump(ret, out_file)


@scorer_group.command()
@click.argument('scorer_dataset_file', type=click.Path(exists=True))
@click.option('-o', '--out-file', type=click.File(mode='w'))
@click.option('--cv', default=10)
@click.option('--k-features', default=None, type=int)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--n-estimators', default=1000)
@click.option('--learning-rate', default=0.01)
@click.option('--max-depth', default=3)
@click.option('--min-samples-split', default=30)
@click.option('--max-features', default=None, type=int)
@click.option('--subsample', default=1.0)
def select_features(scorer_dataset_file, out_file, **kwargs):
    scorer_dataset = joblib.load(scorer_dataset_file)
    ret = scorer.select_features(scorer_dataset, **kwargs)
    if out_file:
        json.dump(ret, out_file, sort_keys=True, indent=2)


@scorer_group.command(name='search_hyper_params')
@click.argument('scorer_dataset_file', type=click.Path(exists=True))
@click.option('-f', '--feature-file', type=click.File())
@click.option('--cv', default=10)
@click.option('--max-evals', default=5000)
@click.option('--n-jobs', default=-1)
@click.option('--n-estimators', default=1000)
@click.option('--random-state', default=0)
def search_scorer_hyper_params(scorer_dataset_file, feature_file, **kwargs):
    scorer_dataset = joblib.load(scorer_dataset_file)
    if feature_file:
        feature_list = json.load(feature_file)['features']
    else:
        feature_list = None

    scorer.search_hyper_params(scorer_dataset, feature_list, **kwargs)


@scorer_group.command()
@click.option('-i', '--input-file', multiple=True, type=click.Path())
@click.option('-o', '--output-dir', type=click.Path())
@click.option('--binary', is_flag=True)
@click.option('--dataset-dir', type=click.Path(exists=True), default='dataset')
@click.option('--entity-db', type=click.Path(), default='entity_db')
def run(dataset_dir, entity_db, **kwargs):
    dataset = DatasetLoader(dataset_dir)
    entity_db = EntityDB.load(entity_db)

    scorer.run(dataset=dataset, entity_db=entity_db, **kwargs)


if __name__ == '__main__':
    cli()

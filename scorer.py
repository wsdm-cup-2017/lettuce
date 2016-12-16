# -*- coding: utf-8 -*-

import click
import itertools
import joblib
import math
import os
import cPickle as pickle
import numpy as np
from collections import defaultdict, Counter, OrderedDict
from contextlib import closing
from functools import partial
from itertools import repeat
from hyperopt import fmin, tpe, hp, STATUS_OK
from keras.preprocessing.sequence import pad_sequences
from multiprocessing.pool import Pool
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score

import classifier_model
import page_classifier
import coocc_classifier
from utils import load_csr, softmax

PAGE_CLF_MODEL_FILES = {
    'pro': [
        ('page_classifier_model_pro_300_full',),
        ('page_classifier_model_pro_300_balanced_full',),
        ('page_classifier_model_pro_attention_300_full',),
        ('page_classifier_model_pro_attention_300_balanced_full',),
        ('page_classifier_model_pro_entity_300_full',),
        ('page_classifier_model_pro_entity_300_balanced_full',),
        ('page_classifier_model_pro_entity_attention_300_full',),
        ('page_classifier_model_pro_entity_attention_300_balanced_full',),
    ],
    'nat': [
        ('page_classifier_model_nat_300_full',),
        ('page_classifier_model_nat_300_balanced_full',),
        ('page_classifier_model_nat_attention_300_full',),
        ('page_classifier_model_nat_attention_300_balanced_full',),
        ('page_classifier_model_nat_entity_300_full',),
        ('page_classifier_model_nat_entity_300_balanced_full',),
        ('page_classifier_model_nat_entity_attention_300_full',),
        ('page_classifier_model_nat_entity_attention_300_balanced_full',),
    ],
}

COOCC_CLF_MODEL_FILES = {
    'pro': [
        ('coocc_classifier_model_pro_win5_300_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_pro_win10_300_full', 'coocc_matrix_win10'),
        ('coocc_classifier_model_pro_win5_300_balanced_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_pro_win10_300_balanced_full', 'coocc_matrix_win10'),
        ('coocc_classifier_model_pro_attention_win5_300_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_pro_attention_win10_300_full', 'coocc_matrix_win10'),
        ('coocc_classifier_model_pro_attention_win5_300_balanced_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_pro_attention_win10_300_balanced_full', 'coocc_matrix_win10'),
    ],
    'nat': [
        ('coocc_classifier_model_nat_win5_300_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_nat_win10_300_full', 'coocc_matrix_win10'),
        ('coocc_classifier_model_nat_win5_300_balanced_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_nat_win10_300_balanced_full', 'coocc_matrix_win10'),
        ('coocc_classifier_model_nat_attention_win5_300_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_nat_attention_win10_300_full', 'coocc_matrix_win10'),
        ('coocc_classifier_model_nat_attention_win5_300_balanced_full', 'coocc_matrix_win5'),
        ('coocc_classifier_model_nat_attention_win10_300_balanced_full', 'coocc_matrix_win10'),
    ]
}

COOCC_MATRICES = ['coocc_matrix_win5', 'coocc_matrix_win10']

CLF_CACHE_FILE = {
    'nat': 'classifier_results_nat.joblib',
    'pro': 'classifier_results_pro.joblib',
}

BINARY_MODEL_FILE = {
    'nat': 'scorer_model_nat_bin.pickle',
    'pro': 'scorer_model_pro_bin.pickle',
}

REGRESSION_MODEL_FILE = {
    'nat': 'scorer_model_nat_reg.pickle',
    'pro': 'scorer_model_pro_reg.pickle',
}

_page_db = None
_entity_db = None
_model_data = {}
_word_occ_matrices = {}
_entity_occ_matrices = {}


def cache_classifier_results(dataset, initial_data, out_file, page_db, entity_db,
                             category, pool_size):
    global _entity_db, _page_db

    if category == 'pro':
        kb_data = dataset.profession_kb
        click.echo('Category: Profession')

    elif category == 'nat':
        kb_data = dataset.nationality_kb
        click.echo('Category: Nationality')

    else:
        raise RuntimeError('Unsupported category: %s' % category)

    _page_db = page_db
    _entity_db = entity_db

    for (clf_name,) in PAGE_CLF_MODEL_FILES[category]:
        with open(clf_name + '.pickle') as f:
            _model_data[clf_name] = pickle.load(f)

    word_coocc_matrices = {}
    entity_coocc_matrices = {}
    for (clf_name, coocc_file) in COOCC_CLF_MODEL_FILES[category]:
        with open(clf_name + '.pickle') as f:
            _model_data[clf_name] = pickle.load(f)

        if coocc_file not in word_coocc_matrices:
            word_coocc_matrices[coocc_file] = load_csr(coocc_file + '_word.npz')
            entity_coocc_matrices[coocc_file] = load_csr(coocc_file + '_entity.npz')

        _word_occ_matrices[clf_name] = word_coocc_matrices[coocc_file]
        _entity_occ_matrices[clf_name] = entity_coocc_matrices[coocc_file]

    target_titles = list(set([title for (title, _) in kb_data]))
    ret = initial_data

    with closing(Pool(pool_size)) as pool:
        for (clf_data, clf_type) in (zip(COOCC_CLF_MODEL_FILES[category], repeat('coocc')) +
                                     zip(PAGE_CLF_MODEL_FILES[category], repeat('page'))):
            clf_name = clf_data[0]
            if clf_name in initial_data:
                continue

            click.echo('Computing predictions using %s' % clf_name)

            model_kwargs = _model_data[clf_name]['model_kwargs']

            feature_options = _model_data[clf_name].get('feature_options', {})

            f = partial(_generate_clf_features,
                        clf_type=clf_type,
                        clf_name=clf_name,
                        text_len=model_kwargs['text_len'],
                        entity_len=model_kwargs['entity_len'],
                        feature_options=feature_options)

            dic = {}
            word_buf = []
            entity_buf = []
            ind = 0
            with click.progressbar(length=len(target_titles)) as bar:
                for item in pool.imap(f, target_titles):
                    if item is not None:
                        title = item[0]
                        word_buf.append(item[1][0])
                        entity_buf.append(item[1][1])
                        dic[title] = ind
                        ind += 1

                    bar.update(1)

            word_buf = np.array(word_buf)
            entity_buf = np.array(entity_buf)

            model = classifier_model.build_model(softmax=False, **model_kwargs)
            model.load_weights(clf_name + '.h5')
            predictions = model.predict([word_buf, entity_buf]).astype('float16')
            del model

            ret[clf_name] = dict(dic=dic, type_list=_model_data[clf_name]['type_list'],
                                 predictions=predictions)

            joblib.dump(ret, out_file)


def _generate_clf_features(title, clf_name, clf_type, feature_options, text_len,
                           entity_len):
    title = _entity_db.resolve_redirect(title)
    word_dic = _model_data[clf_name]['word_dic']
    entity_dic = _model_data[clf_name]['entity_dic']

    if clf_type == 'page':
        feat_obj = page_classifier.generate_features(
            title, _page_db, word_dic, entity_dic, **feature_options
        )
    elif clf_type == 'coocc':
        feat_obj = coocc_classifier.generate_features(
            title, _word_occ_matrices[clf_name], _entity_occ_matrices[clf_name], word_dic, entity_dic
        )
    else:
        raise RuntimeError()

    if feat_obj is not None:
        return (title, (pad_sequences([feat_obj[0]], maxlen=text_len, dtype='int32')[0],
                        pad_sequences([feat_obj[1]], maxlen=entity_len, dtype='int32')[0]))


def build_dataset(dataset, clf_cache, entity_db, category, binary=False,
                  target_data=None):
    if category == 'pro':
        kb_data = dataset.profession_kb
        if target_data is None:
            target_data = dataset.profession_train
        click.echo('Category: Profession')

    elif category == 'nat':
        kb_data = dataset.nationality_kb
        if target_data is None:
            target_data = dataset.nationality_train
        click.echo('Category: Nationality')

    # compute scores for PMI
    multi_type_kb_data = [types for (_, types) in kb_data if len(types) != 1]
    type_counter = Counter()
    for types in multi_type_kb_data:
        for type_name in types:
            type_counter[type_name] += 1
        for comb in itertools.combinations(types, 2):
            type_counter[tuple(sorted(comb))] += 1

    type_proba = defaultdict(lambda: 1.0 / len(multi_type_kb_data))
    for (key, count) in type_counter.items():
        type_proba[key] = float(count + 1) / len(multi_type_kb_data)

    titles = []
    type_names = []
    features = []
    labels = []

    with click.progressbar(target_data) as bar:
        for (n, (title, type_score_pairs)) in enumerate(bar):
            title = entity_db.resolve_redirect(title)
            predicted_values = {}
            predicted_probas = {}
            valid_values = {}
            valid_probas = {}
            predicted_type = {}
            valid_types = frozenset(o[0] for o in type_score_pairs)

            for clf_model_name in clf_cache.keys():
                cache_obj = clf_cache[clf_model_name]
                index = cache_obj['dic'].get(title)
                if index is None:
                    continue

                type_list = cache_obj['type_list']
                values = cache_obj['predictions'][index].astype('float32')
                probas = softmax(values)

                predicted_values[clf_model_name] = {t: v for (t, v) in zip(type_list, values)}
                predicted_probas[clf_model_name] = {t: v for (t, v) in zip(type_list, probas)}

                valid_values[clf_model_name] = {
                    t: v for (t, v) in predicted_values[clf_model_name].items() if t in valid_types
                }
                valid_probas[clf_model_name] = {
                    t: v for (t, v) in predicted_probas[clf_model_name].items() if t in valid_types
                }
                predicted_type[clf_model_name] = sorted(valid_values[clf_model_name].items(), key=lambda o: o[1])[-1][0]

            num_types = len(type_score_pairs)

            for (type_name, score) in type_score_pairs:
                if (score is not None) and binary and (3 <= score <= 4):
                    continue

                feat = {}

                feat['num_types'] = num_types

                for clf_model_name in clf_cache.keys():
                    if clf_model_name not in predicted_probas:
                        continue

                    if len(valid_probas[clf_model_name].keys()) == 0:
                        continue

                    proba = predicted_probas[clf_model_name].get(type_name, 0.0)
                    raw_value = predicted_values[clf_model_name].get(type_name, 0.0)
                    feat[clf_model_name + '_proba'] = proba
                    feat[clf_model_name + '_raw_value'] = raw_value
                    feat[clf_model_name + '_top_proba_diff'] = np.max(valid_probas[clf_model_name].values()) - proba
                    feat[clf_model_name + '_bottom_proba_diff'] = np.min(valid_probas[clf_model_name].values()) - proba
                    feat[clf_model_name + '_top_diff'] = np.max(valid_values[clf_model_name].values()) - raw_value
                    feat[clf_model_name + '_bottom_diff'] = np.min(valid_values[clf_model_name].values()) - raw_value

                    feat[clf_model_name + '_pmi'] = (
                        type_proba[tuple(sorted([type_name, predicted_type[clf_model_name]]))] /
                        (type_proba[type_name] * type_proba[predicted_type[clf_model_name]])
                    )

                    for (feat_key, feat_val) in feat.items():
                        feat_val = float(feat_val)
                        if math.isnan(feat_val):
                            feat[feat_key] = 0.0
                        elif feat_val >= np.finfo('float32').max:
                            feat[feat_key] = float(np.finfo('float32').max) - 1
                        elif feat_val <= np.finfo('float32').min:
                            feat[feat_key] = float(np.finfo('float32').min) + 1

                titles.append(title)
                type_names.append(type_name)
                features.append(feat)
                if binary and (score is not None):
                    labels.append(bool(score >= 5))
                else:
                    labels.append(score)

    return dict(titles=titles, type_names=type_names, features=features,
                labels=labels, binary=binary)


def train_model(scorer_dataset, feature_list, **model_kwargs):
    vectorizer = DictVectorizer(sparse=False)

    features = scorer_dataset['features']
    if feature_list:
        feature_list = frozenset(feature_list)
        features = [{k: v for (k, v) in f.items() if k in feature_list} for f in features]

    mat = vectorizer.fit_transform(features)
    labels = np.array(scorer_dataset['labels'])
    binary = scorer_dataset['binary']

    if binary:
        model = GradientBoostingClassifier(random_state=0, **model_kwargs)

    else:
        model = GradientBoostingRegressor(random_state=0, **model_kwargs)

    model = model.fit(mat, labels)
    return dict(model=model, vectorizer=vectorizer)


def evaluate(scorer_dataset, feature_list, cv, **model_kwargs):
    vectorizer = DictVectorizer(sparse=False)

    features = scorer_dataset['features']
    if feature_list:
        feature_list = frozenset(feature_list)
        features = [{k: v for (k, v) in f.items() if k in feature_list} for f in features]

    mat = vectorizer.fit_transform(features)
    labels = np.array(scorer_dataset['labels'])
    binary = scorer_dataset['binary']

    if binary:
        model = GradientBoostingClassifier(random_state=0, **model_kwargs)

    else:
        model = GradientBoostingRegressor(random_state=0, **model_kwargs)

    kf = KFold(n_splits=cv, shuffle=True, random_state=0)

    predicted = []
    gs = []
    for (train_indices, test_indices) in kf.split(mat):
        model = model.fit(mat[train_indices], labels[train_indices])
        predicted.append(model.predict(mat[test_indices]))
        gs.append(labels[test_indices])

    if binary:
        accuracy = np.mean([accuracy_score(g, p) for (g, p) in zip(predicted, gs)])
        click.echo('Accuracy: %.3f' % accuracy)

    else:
        mae = []
        mae_raw = []
        delta_1 = []
        delta_2 = []
        delta_4 = []
        for (pred, scores) in zip(predicted, gs):
            mae_raw.append(mean_absolute_error(pred, scores))
            pred = [round(p) for p in pred]
            mae.append(mean_absolute_error(pred, scores))

            abs_diff = np.array([abs(s - p) for (s, p) in zip(scores, pred)])
            delta_1.append(float(np.where(abs_diff <= 1)[0].shape[0]) / len(abs_diff))
            delta_2.append(float(np.where(abs_diff <= 2)[0].shape[0]) / len(abs_diff))
            delta_4.append(float(np.where(abs_diff <= 4)[0].shape[0]) / len(abs_diff))

        click.echo('Mean absolute error: %.3f' % np.mean(mae))
        click.echo('Mean absolute error (raw): %.3f' % np.mean(mae_raw))
        click.echo('Delta 1: %.3f' % np.mean(delta_1))
        click.echo('Delta 2: %.3f' % np.mean(delta_2))
        click.echo('Delta 4: %.3f' % np.mean(delta_4))


def run(input_file, output_dir, binary, dataset, entity_db):
    for in_file in input_file:
        click.echo('Input file: %s' % in_file)

        with open(in_file) as f:
            target_data = OrderedDict()
            for line in f:
                items = line.rstrip().decode('utf-8').split('\t')
                title = items[0]
                type_name = items[1]
                if title not in target_data:
                    target_data[title] = []

                target_data[title].append((type_name, None))

            target_data = target_data.items()

        if 'nationality' in os.path.basename(in_file):
            category = 'nat'
        else:
            category = 'pro'

        if binary:
            model_file = BINARY_MODEL_FILE[category]
        else:
            model_file = REGRESSION_MODEL_FILE[category]

        clf_cache = joblib.load(CLF_CACHE_FILE[category], mmap_mode='r')
        with open(model_file) as f:
            model = pickle.load(f)

        ds = build_dataset(dataset, clf_cache, entity_db, category=category,
                           target_data=target_data)
        vectorizer = model['vectorizer']
        mat = vectorizer.transform(ds['features'])

        predictions = iter(model['model'].predict(mat))

        out_file = os.path.join(output_dir, os.path.basename(in_file))
        click.echo('Output file: %s' % out_file)

        with open(out_file, 'w') as f:
            for (title, type_score_pairs) in target_data:
                for (type_name, _) in type_score_pairs:
                    prediction = predictions.next()
                    if binary:
                        if prediction:
                            value = 5
                        else:
                            value = 2
                    else:
                        value = int(round(prediction))
                        value = min(7, value)
                        value = max(0, value)

                    f.write('%s\t%s\t%d\n' % (title.encode('utf-8'), type_name.encode('utf-8'), value))


def select_features(scorer_dataset, cv, k_features, pool_size, **model_kwargs):
    vectorizer = DictVectorizer(sparse=False)
    features = scorer_dataset['features']
    mat = vectorizer.fit_transform(features)
    labels = np.array(scorer_dataset['labels'])

    if scorer_dataset['binary']:
        model = GradientBoostingClassifier(random_state=0, **model_kwargs)
        scoring = 'accuracy'
    else:
        model = GradientBoostingRegressor(random_state=0, **model_kwargs)
        scoring = 'neg_mean_absolute_error'

    if k_features is None:
        k_features = mat.shape[1]

    active_indices = set()
    history = []
    f = partial(_compute_cv_score, mat=mat, labels=labels, model=model, cv=cv,
                scoring=scoring)

    with closing(Pool(pool_size)) as pool:
        for n in range(k_features):
            indices = [list(active_indices) + [ind] for ind in range(mat.shape[1])
                       if ind not in active_indices]
            scores = pool.map(f, indices)
            selected_index = list(set(indices[np.argmax(scores)]) - active_indices)[0]

            active_indices.add(selected_index)
            history.append(dict(
                score=max(scores),
                features=[vectorizer.feature_names_[i] for i in active_indices],
            ))
            if len(history) == 1:
                score_diff = 0.0
            else:
                score_diff = history[-1]['score'] - history[-2]['score']

            click.echo('%d: %.3f (%.3f): +%s' %
                       (n, history[-1]['score'], score_diff, vectorizer.feature_names_[selected_index]))

    return sorted(history, key=lambda o: o['score'])[-1]


def _compute_cv_score(indices, mat, labels, model, cv, **kwargs):
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    if model.max_features and model.max_features > len(indices):
        model.max_features = len(indices)

    return np.mean(cross_val_score(model, mat[:, indices], labels, cv=kf, **kwargs))


def search_hyper_params(scorer_dataset, feature_list, cv, max_evals, n_jobs, **model_kwargs):
    vectorizer = DictVectorizer(sparse=False)

    features = scorer_dataset['features']
    if feature_list:
        feature_list = frozenset(feature_list)
        features = [{k: v for (k, v) in f.items() if k in feature_list} for f in features]

    mat = vectorizer.fit_transform(features)
    labels = np.array(scorer_dataset['labels'])

    search_space = {
        'subsample':  hp.quniform('subsample', 0.05, 1.0, 0.05),
        'min_samples_split': hp.quniform('min_samples_split', 2, 100, 1),
        'max_depth': hp.choice('max_depth', range(1, 7)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 0.005),
        'max_features': hp.quniform('max_features', 3, mat.shape[1], 1),
    }
    binary = scorer_dataset['binary']

    best_score = [-float('inf')]
    eval_count = [0]

    def f(params, **model_kwargs):
        click.echo('eval: %d/%d' % (eval_count[0], max_evals))
        eval_count[0] += 1

        params['min_samples_split'] = int(params['min_samples_split'])
        params['max_depth'] = int(params['max_depth'])
        params['max_features'] = int(params['max_features'])

        click.echo(params)

        model_kwargs.update(params)

        if binary:
            model = GradientBoostingClassifier(**model_kwargs)
            scoring = 'accuracy'
        else:
            model = GradientBoostingRegressor(**model_kwargs)
            scoring = 'neg_mean_absolute_error'

        kf = KFold(n_splits=cv, shuffle=True, random_state=0)
        score = np.mean(cross_val_score(model, mat, labels, cv=kf, scoring=scoring, n_jobs=n_jobs))

        if score > best_score[0]:
            click.secho('score: %.3f' % score, fg='green')
            best_score[0] = score
        elif score == best_score[0]:
            click.secho('score: %.3f' % score, fg='yellow')
        else:
            click.echo('score: %.3f (best: %.3f)' % (score, best_score[0]))

        return {'loss': -score, 'status': STATUS_OK}

    target_func = partial(f, **model_kwargs)

    best = fmin(fn=target_func, space=search_space, algo=tpe.suggest,
                max_evals=max_evals)

    click.echo('best score: %.3f' % best_score[0])
    click.echo('params: %s' % best)

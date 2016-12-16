# -*- coding: utf-8 -*-

import click
import random
import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight

from utils.tokenizer import DefaultTokenizer

_tokenizer = DefaultTokenizer()


def build_dataset(dataset, page_db, entity_db, category, text_len, min_text_len,
                  entity_len, min_entity_len, abstract_only, random_seed,
                  min_word_count, min_entity_count, in_links, out_links,
                  dev_size, test_size, **kwargs):
    random.seed(random_seed)
    np.random.seed(random_seed)

    if category == 'pro':
        kb_data = dataset.profession_kb
        click.echo('Category: Profession')

    elif category == 'nat':
        kb_data = dataset.nationality_kb
        click.echo('Category: Nationality')

    else:
        raise RuntimeError('Unsupported category: %s' % category)

    target_data = [(title, types[0]) for (title, types) in kb_data
                   if len(types) == 1]

    word_counter = Counter()
    entity_counter = Counter()

    click.echo('Building vocabulary...')

    with click.progressbar(page_db.keys()) as bar:
        for key in bar:
            title = key.decode('utf-8')
            title = entity_db.resolve_redirect(title)

            words = _tokenizer.tokenize(page_db.get_text(title, abstract_only))
            words = words[:text_len]
            entities = set([l.title for l in page_db.get_links(title, in_links, out_links, abstract_only)])

            word_counter.update(w.lower() for w in words)
            entity_counter.update(entities)

    word_dic = {
        k: n for (n, k) in enumerate([w for (w, c) in word_counter.iteritems() if c >= min_word_count], 1)
    }
    entity_dic = {
        k: n for (n, k) in enumerate([e for (e, c) in entity_counter.iteritems() if c >= min_entity_count], 1)
    }

    click.echo('Building dataset...')

    type_list = list(set([t for (_, t) in target_data]))
    type_index = {t: n for (n, t) in enumerate(type_list)}

    words_arr = []
    entities_arr = []
    label_names = []
    titles = []
    feature_options = dict(in_links=in_links, out_links=out_links,
                           abstract_only=abstract_only)
    with click.progressbar(target_data) as bar:
        for (n, (title, type_name)) in enumerate(bar):
            title = entity_db.resolve_redirect(title)
            ret = generate_features(title, page_db, word_dic, entity_dic,
                                    **feature_options)
            if ret is None:
                continue

            (word_f, entity_f) = ret

            if len(word_f) <= min_text_len:
                continue
            if len(entity_f) <= min_entity_len:
                continue

            words_arr.append(word_f)
            entities_arr.append(entity_f)
            label_names.append(type_name)
            titles.append(title)

    titles = np.array(titles)
    words_arr = pad_sequences(words_arr, maxlen=text_len, dtype='int32')
    entities_arr = pad_sequences(entities_arr, maxlen=entity_len, dtype='int32')

    labels = np.zeros((len(label_names), len(type_list)), dtype='int32')
    for (n, label_name) in enumerate(label_names):
        labels[n][type_index[label_name]] = 1

    dev_count = int(float(len(words_arr)) * dev_size)
    click.echo('dev size: %d' % dev_count)
    test_count = int(float(len(words_arr)) * test_size)
    click.echo('test size: %d' % test_count)

    indices = set(range(len(words_arr)))
    data = {}
    titles_data = {}

    if dev_count > 0:
        dev_indices = random.sample(indices, dev_count)
        data['dev'] = (words_arr[dev_indices], entities_arr[dev_indices],
                       labels[dev_indices])
        titles_data['dev'] = titles[dev_indices]
        indices -= set(dev_indices)

    if test_count > 0:
        test_indices = random.sample(indices, test_count)
        data['test'] = (words_arr[test_indices], entities_arr[test_indices],
                        labels[test_indices])
        titles_data['test'] = titles[test_indices]
        indices -= set(test_indices)

    train_indices = list(indices)
    data['train'] = (words_arr[train_indices], entities_arr[train_indices],
                     labels[train_indices])
    titles_data['train'] = titles[train_indices]

    class_weight = compute_class_weight('balanced', np.array(type_list),
                                        [t for (_, t) in target_data])
    class_weight = {i: w for (i, w) in enumerate(class_weight)}

    return dict(data=data, titles=titles_data, type_list=type_list,
                word_dic=word_dic, entity_dic=entity_dic, category=category,
                class_weight=class_weight, feature_options=feature_options)


def generate_features(title, page_db, word_dic, entity_dic, in_links=True,
                      out_links=True, abstract_only=False):
    try:
        text = page_db.get_text(title, abstract_only)
        links = set([l.title for l in page_db.get_links(title, in_links, out_links, abstract_only)])

    except KeyError:
        return None

    word_f = [word_dic[w.lower()] for w in _tokenizer.tokenize(text) if w.lower() in word_dic]
    entity_f = [entity_dic[e] for e in links if e in entity_dic]

    return (word_f, entity_f)

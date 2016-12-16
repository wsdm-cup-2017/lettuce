# -*- coding: utf-8 -*-

import click
import itertools
import random
import re
import cPickle as pickle
import numpy as np
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from scipy.sparse import lil_matrix
from sklearn.utils.class_weight import compute_class_weight

from utils.tokenizer import DefaultTokenizer
from utils import load_csr, save_csr

LINK_RE = re.compile(ur'\[(.+?)\|(.*?)\]')
WIKI_SENTENCES_LINE_LEN = 33159353

_tokenizer = DefaultTokenizer()


def build_coocc_matrix(sentence_db, out_file, word_window, min_word_count):
    word_counter = Counter()
    entity_set = set()

    click.echo('Step 1/2: building vocabulary...')

    with click.progressbar(sentence_db.itervalues(),
                           length=len(sentence_db)) as bar:
        for sent in bar:
            words = _tokenizer.tokenize(sent.text)
            word_counter.update(w.lower() for w in words)

            for link in sent.wiki_links:
                entity_set.add(link.title)

    word_dic = {
        k: n for (n, k) in enumerate([w for (w, c) in word_counter.iteritems()
                                      if c >= min_word_count], 1)
    }
    entity_dic = {k: n for (n, k) in enumerate(entity_set, 1)}

    click.echo('The number of words: %d' % len(word_dic))

    word_counter = None
    entity_set = None

    click.echo('Step 2/2: building matrix...')

    word_occ_matrix = lil_matrix((len(entity_dic) + 1, len(word_dic) + 1), dtype='uint16')
    entity_occ_matrix = lil_matrix((len(entity_dic) + 1, len(entity_dic) + 1), dtype='uint16')

    with click.progressbar(sentence_db.itervalues(),
                           length=len(sentence_db)) as bar:
        for sent in bar:
            text = sent.text.lower()
            span_word_pairs = [(s, text[s[0]:s[1]]) for s in _tokenizer.span_tokenize(text)]
            span_index_pairs = [(s, word_dic.get(w)) for (s, w) in span_word_pairs]
            entity_occs = set()

            for link in sent.wiki_links:
                entity_index = entity_dic[link.title]
                entity_occs.add(entity_index)

                c = 0
                for (span, index) in reversed(span_index_pairs):
                    if span[1] <= link.span[0]:
                        words.append(text[span[0]:span[1]])
                        if index is not None:
                            word_occ_matrix[entity_index, index] += 1
                        c += 1
                        if word_window and (c == word_window):
                            break

                c = 0
                for (span, index) in span_index_pairs:
                    if span[0] >= link.span[1]:
                        words.append(text[span[0]:span[1]])
                        if index is not None:
                            word_occ_matrix[entity_index, index] += 1
                        c += 1
                        if word_window and (c == word_window):
                            break

            for (ind1, ind2) in itertools.combinations(entity_occs, 2):
                entity_occ_matrix[ind1, ind2] += 1
                entity_occ_matrix[ind2, ind1] += 1

    word_occ_matrix = word_occ_matrix.tocsr()
    save_csr(out_file + '_word.npz', word_occ_matrix)

    entity_occ_matrix = entity_occ_matrix.tocsr()
    save_csr(out_file + '_entity.npz', entity_occ_matrix)

    with open(out_file + '_vocab.pickle', 'w') as f:
        pickle.dump(dict(word_dic=word_dic, entity_dic=entity_dic), f)


def build_dataset(dataset, entity_db, coocc_matrix_file, category, text_len,
                  entity_len, min_text_len, min_entity_len, random_seed,
                  dev_size, test_size):
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

    type_list = list(set([t for (_, t) in target_data]))
    type_index = {t: n for (n, t) in enumerate(type_list)}

    word_occ_matrix = load_csr(coocc_matrix_file + '_word.npz')
    entity_occ_matrix = load_csr(coocc_matrix_file + '_entity.npz')
    with open(coocc_matrix_file + '_vocab.pickle') as f:
        data = pickle.load(f)
        word_dic = data['word_dic']
        entity_dic = data['entity_dic']

    words_arr = []
    entities_arr = []
    label_names = []
    titles = []
    with click.progressbar(target_data) as bar:
        for (n, (title, type_name)) in enumerate(bar):
            title = entity_db.resolve_redirect(title)
            if title not in entity_dic:
                continue

            ret = generate_features(title, word_occ_matrix, entity_occ_matrix,
                                    word_dic, entity_dic)
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
                class_weight=class_weight)


def generate_features(title, word_occ_matrix, entity_occ_matrix, word_dic,
                      entity_dic):
    try:
        entity_index = entity_dic[title]
    except KeyError:
        return None

    word_row = word_occ_matrix[entity_index]
    entity_row = entity_occ_matrix[entity_index]

    word_f = list(itertools.chain(*[[i] * word_row[0, i] for i in word_row.nonzero()[1]]))
    word_f = np.random.permutation(word_f)

    entity_f = list(itertools.chain(*[[i] * entity_row[0, i] for i in entity_row.nonzero()[1]]))
    entity_f = np.random.permutation(entity_f)

    return (word_f, entity_f)

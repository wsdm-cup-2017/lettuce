# -*- coding: utf-8 -*-

import click
import random
import theano
import cPickle as pickle
import keras.backend as K
import numpy as np
from functools import partial
from hyperopt import fmin, tpe, hp, STATUS_OK
from keras.backend.common import floatx
from keras.layers import Input, Activation, Dense, Dropout, Layer, Lambda, TimeDistributed, Reshape, merge
from keras.layers.embeddings import Embedding
from keras.models import Model
from theano import tensor as T

FLOATX = floatx()


class WeightedAverageLayer(Layer):
    def call(self, inputs, mask=None):
        def f(i, embedding, text_input, weights):
            mask = T.neq(text_input[i], 0).astype(FLOATX)
            weighted = mask * weights[i]
            vec = T.dot(weighted, embedding[i])
            vec /= T.maximum(vec.norm(2, 0), K.epsilon())

            return vec

        return theano.map(f, T.arange(inputs[0].shape[0]), non_sequences=inputs)[0]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])


def search_hyper_params(dataset, max_evals, **kwargs):
    dropout_keep_probs = [0.05 * n for n in range(1, 21)]
    hidden_units_list = [500 * n for n in range(1, 21)]
    dim_sizes = [100, 150, 200]
    attention_dim_sizes = [2 * n for n in range(1, 11)]

    search_space = {
        'dropout_keep_prob': hp.quniform('dropout_keep_prob', 0, len(dropout_keep_probs) - 1, 1),
        'hidden_units':  hp.quniform('hidden_units', 0, len(hidden_units_list) - 1, 1),
        'dim_size':  hp.quniform('dim_size', 0, len(dim_sizes) - 1, 1),
        'attention_dim_size':  hp.quniform('attention_dim_size', 0, len(attention_dim_sizes) - 1, 1),
    }
    search_history = []
    best_val_acc = [0.0]

    def f(params, **train_kwargs):
        kwargs = {}
        kwargs['dropout_keep_prob'] = dropout_keep_probs[int(params['dropout_keep_prob'])]
        kwargs['hidden_units'] = hidden_units_list[int(params['hidden_units'])]
        kwargs['dim_size'] = dim_sizes[int(params['dim_size'])]
        kwargs['attention_dim_size'] = attention_dim_sizes[int(params['attention_dim_size'])]

        click.echo(kwargs)

        train_kwargs.update(kwargs)

        history = train_model(dataset, out_file=None, **train_kwargs)['history']

        max_val_acc = max(history['val_acc'])
        if search_history:
            best_val_acc[0] = sorted(search_history)[-1][0]

        if max_val_acc > best_val_acc[0]:
            click.secho('max_val_acc: %.3f' % max_val_acc, fg='green')
        elif max_val_acc == best_val_acc[0]:
            click.secho('max_val_acc: %.3f' % max_val_acc, fg='yellow')
        else:
            click.echo('max_val_acc: %.3f' % max_val_acc)

        search_history.append((max_val_acc, kwargs))

        return {'loss': min(history['val_loss']), 'status': STATUS_OK}

    target_func = partial(f, **kwargs)

    best = fmin(fn=target_func, space=search_space, algo=tpe.suggest, max_evals=max_evals)
    print best


def train_model(dataset, out_file, batch_size, epoch, balanced_weight,
                random_seed, temp_dir='/run/shm', **model_kwargs):
    random.seed(random_seed)
    np.random.seed(random_seed)

    num_classes = len(dataset['type_list'])
    word_dic = dataset['word_dic']
    entity_dic = dataset['entity_dic']

    model_kwargs.update(dict(
        num_classes=num_classes,
        word_size=len(word_dic) + 1,
        entity_size=len(entity_dic) + 1,
    ))

    model = build_model(**model_kwargs)

    fit_kwargs = {}
    if 'dev' in dataset['data']:
        dev_data = list(generate_data(dataset['data']['dev'], num_classes, batch_size, loop=False))
        dev_data = ([np.vstack([d[0][0] for d in dev_data]),
                     np.vstack([d[0][1] for d in dev_data])],
                    np.vstack([d[1] for d in dev_data]))
        fit_kwargs['validation_data'] = dev_data

    else:
        dev_data = None

    if balanced_weight:
        class_weight = dataset['class_weight']
        fit_kwargs['class_weight'] = {i: float(w) / max(class_weight.values())
                                      for (i, w) in class_weight.items()}

    history = model.fit_generator(generate_data(dataset['data']['train'], num_classes, batch_size),
                                  samples_per_epoch=dataset['data']['train'][0].shape[0],
                                  nb_epoch=epoch,
                                  max_q_size=1000,
                                  **fit_kwargs)

    if out_file is not None:
        model.save_weights(out_file + '.h5')
        with open(out_file + '.pickle', 'w') as f:
            pickle.dump(dict(
                model_kwargs=model_kwargs,
                word_dic=word_dic,
                entity_dic=entity_dic,
                type_list=dataset['type_list'],
                category=dataset['category'],
                feature_options=dataset.get('feature_options'),
            ), f, protocol=-1)

    if 'test' in dataset['data']:
        test_data = list(generate_data(dataset['data']['test'], num_classes, batch_size, loop=False))
        test_data = ([np.vstack([d[0][0] for d in test_data]),
                      np.vstack([d[0][1] for d in test_data])],
                     np.vstack([d[1] for d in test_data]))

        click.echo('\nTest accuracy: %.3f' % model.evaluate(test_data[0], test_data[1])[1])

    return dict(model=model, history=history.history)


def build_model(text_len, entity_len, num_classes, optimizer, word_size, entity_size,
                dim_size, word_only, entity_only, attention, attention_dim_size,
                hidden_units, dropout_keep_prob, softmax=True, **kwargs):
    text_input_layer = Input(shape=(text_len,), dtype='int32')

    if not entity_only:
        word_embed_layer = Embedding(
            word_size, dim_size, input_length=text_len, name='word_embedding',
        )(text_input_layer)

        if attention:
            word_attention_embed_layer = Embedding(
                word_size, attention_dim_size, input_length=text_len,
                name='word_attention_embedding',
            )(text_input_layer)

            word_attention_layer = TimeDistributed(Dense(1))(word_attention_embed_layer)
            word_attention_layer = Reshape((text_len,))(word_attention_layer)
            word_attention_layer = Activation('softmax')(word_attention_layer)

        else:
            word_attention_layer = Lambda(lambda x: T.ones(x.shape))(text_input_layer)

        text_layer = WeightedAverageLayer(name='text_layer')(
            [word_embed_layer, text_input_layer, word_attention_layer]
        )

    entity_input_layer = Input(shape=(entity_len,), dtype='int32')

    if not word_only:
        entity_embed_layer = Embedding(
            entity_size, dim_size, input_length=entity_len, name='entity_embedding',
        )(entity_input_layer)

        if attention:
            entity_attention_embed_layer = Embedding(
                entity_size, attention_dim_size, input_length=entity_len,
                name='entity_attention_embedding',
            )(entity_input_layer)

            entity_attention_layer = TimeDistributed(Dense(1))(entity_attention_embed_layer)
            entity_attention_layer = Reshape((entity_len,))(entity_attention_layer)
            entity_attention_layer = Activation('softmax')(entity_attention_layer)

        else:
            entity_attention_layer = Lambda(lambda x: T.ones(x.shape))(entity_input_layer)

        entity_layer = WeightedAverageLayer(name='entity_layer')(
            [entity_embed_layer, entity_input_layer, entity_attention_layer]
        )

    if word_only:
        combine_layer = text_layer
    elif entity_only:
        combine_layer = entity_layer
    else:
        combine_layer = merge([text_layer, entity_layer], mode='concat', concat_axis=-1)

    hidden_layer = Dense(hidden_units)(combine_layer)
    hidden_layer = Activation('relu')(hidden_layer)
    hidden_layer = Dropout(dropout_keep_prob)(hidden_layer)

    output_layer = Dense(num_classes)(hidden_layer)
    predictions = Activation('softmax')(output_layer)

    if softmax:
        model = Model(input=[text_input_layer, entity_input_layer],
                      output=predictions)
    else:
        model = Model(input=[text_input_layer, entity_input_layer],
                      output=output_layer)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def generate_data(data, num_classes, batch_size, loop=True):
    (words_arr, entities_arr, labels_arr) = data

    while True:
        word_buf = []
        entity_buf = []
        label_buf = []
        for n in range(words_arr.shape[0]):
            word_buf.append(words_arr[n])
            entity_buf.append(entities_arr[n])
            label_buf.append(labels_arr[n])

            if len(word_buf) == batch_size:
                yield ([[np.array(word_buf), np.array(entity_buf)], np.array(label_buf)])

                word_buf = []
                entity_buf = []
                label_buf = []

        yield ([[np.array(word_buf), np.array(entity_buf)], np.array(label_buf)])

        if not loop:
            break

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 11:15
# @Author  : Flavorfan
# @File    : seq2seq_equations.py
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, GRU
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model



def generate_equations(allowed_operators, dataset_size, min_value, max_value):
    sample = []
    for _ in range(dataset_size):
        a = random.randint(min_value, max_value)
        b = random.randint(min_value, max_value)
        op = random.choice(allowed_operators)
        eqution = str(a) + op + str(b)
        solution = str(eval(eqution))
        sample.append((eqution, solution))
    return sample

def test_generate_equations():
    allowed_operators = ['+', '-']
    dataset_size = 10
    for (input_, output_) in generate_equations(allowed_operators, dataset_size, 0, 100):
        if not (type(input_) is str and type(output_) is str):
            return "Both parts should be strings."
        if eval(input_) != int(output_):
            return "The (equation: {!r}, solution: {!r}) pair is incorrect.".format(input_, output_)
    return "Tests passed."

def gen_data():
    allowed_operators = ['+', '-']
    dataset_size = 100000
    data = generate_equations(allowed_operators, dataset_size, min_value=0, max_value=9999)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    return train_set, test_set

def sentence_to_ids(sentence, word2id, padded_len):
    sent_len = min(len(sentence) + 1, padded_len)
    plen = max(0, padded_len - sent_len)
    sent_ids = [word2id[word] for word in sentence[:sent_len - 1]] + [word2id[end_symbol]] + [
        word2id[padding_symbol]] * plen
    return sent_ids, sent_len

def test_sentence_to_ids():
    sentences = [("123+123", 7), ("123+123", 8), ("123+123", 10)]
    expected_output = [([5, 6, 7, 3, 5, 6, 2], 7),
                       ([5, 6, 7, 3, 5, 6, 7, 2], 8),
                       ([5, 6, 7, 3, 5, 6, 7, 2, 0, 0], 8)]
    for (sentence, padded_len), (sentence_ids, expected_length) in zip(sentences, expected_output):
        output, length = sentence_to_ids(sentence, word2id, padded_len)
        if output != sentence_ids:
            return("Convertion of '{}' for padded_len={} to {} is incorrect.".format(
                sentence, padded_len, output))
        if length != expected_length:
            return("Convertion of '{}' for padded_len={} has incorrect actual length {}.".format(
                sentence, padded_len, length))
    return("Tests passed.")


def batch_to_ids(sentences, word2id, max_len):
    max_len_in_batch = min(max(len(s) for s in sentences) + 1, max_len)
    batch_ids, batch_ids_len = [], []
    for sentence in sentences:
        ids, ids_len = sentence_to_ids(sentence, word2id, max_len_in_batch)
        batch_ids.append(ids)
        batch_ids_len.append(ids_len)
    return batch_ids, batch_ids_len

def generate_batches(samples, batch_size=64):
    X, Y = [], []
    for i, (x, y) in enumerate(samples, 1):
        X.append(x)
        Y.append(y)
        if i % batch_size == 0:
            yield X, Y
            X, Y = [], []
    if X and Y:
        yield X, Y

class Seq2SeqModel(object):
    # def __create_embeddings(self, vocab_size,embedding_dim, batch_size):
    #     self.input_batch_embedded  = tf.keras.layers.Embedding(
    #                                         vocab_size, embedding_dim,
    #                                         batch_input_shape=[batch_size, None])
    # def __build_encoder(self, rnn_units):
    #     _,self.final_encoder_state = tf.keras.layers.GRU(rnn_units, # whole_sequence_output, final_state = gru(inputs)
    #                             return_sequences=True,
    #                             # stateful=True,
    #                             return_state=True,
    #                             # recurrent_initializer='glorot_uniform'
    #                             )
    #
    # def __build_decoder(self, hidden_size, vocab_size, max_iter, start_symbol_id, end_symbol_id):
    #     # Use start symbols as the decoder inputs at the first time step.
    #     batch_size = tf.shape(self.input_batch)[0]
    #     start_tokens = tf.fill([batch_size], start_symbol_id)
    #
    #     ground_truth_as_input = tf.concat([tf.expand_dims(start_tokens, 1), self.ground_truth], 1)
    @staticmethod
    def build(vocab_size, embeddings_size, hidden_size,batch_size,
              max_iter, start_symbol_id, end_symbol_id, padding_symbol_id):

        encoder_inputs = Input(shape=(batch_size, None))
        encoder_embedding = Embedding(vocab_size,embeddings_size)(encoder_inputs)
        encoder_outputs, final_encoder_state = GRU(hidden_size,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   recurrent_initializer='glorot_uniform')(encoder_embedding)

        # build the encoder model
        encoder = Model(encoder_inputs, (encoder_outputs, final_encoder_state), name="encoder")
        print(encoder.summary())

        # decoder
        decoder_inputs = Input(shape=(batch_size, None))
        decoder_embeddings = Embedding(vocab_size,embeddings_size)(decoder_inputs)
        decoder_outputs, decoder_state = GRU(hidden_size,
                                                   return_sequences=True,
                                                   return_state=True,
                                                   recurrent_initializer='glorot_uniform')(decoder_embeddings)

        decoder = Model(decoder_inputs,(decoder_outputs, decoder_state), name='decoder')
        print(decoder.summary())

        autoencoder = Model()







if __name__ == '__main__':
    # print(test_generate_equations())

    train_set, test_set = gen_data()

    word2id = {symbol: i for i, symbol in enumerate('#^$+-1234567890')}
    id2word = {i: symbol for symbol, i in word2id.items()}

    # special symbols
    start_symbol = '^'
    end_symbol = '$'
    padding_symbol = '#'

    # print(test_sentence_to_ids())  # padding

    # test batch to ids
    sentences = train_set[0]
    ids, sent_lens = batch_to_ids(sentences, word2id, max_len=10)
    print('Input:', sentences)
    print('Ids: {}\nSentences lengths: {}'.format(ids, sent_lens))


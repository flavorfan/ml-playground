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
import os
import time

word2id = {symbol: i for i, symbol in enumerate('#^$+-1234567890')}
id2word = {i: symbol for symbol, i in word2id.items()}
# special symbols
start_symbol = '^'
end_symbol = '$'
padding_symbol = '#'


checkpoint_dir = './ckpts/seq2seq_equiations'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

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


def generate_batch_dataset(data,bath_size = 128, max_len=10):
    buffer_size = len(data)

    X_str, Y_str = [], []
    for i, (x, y) in enumerate(data):
        X_str.append(x)
        Y_str.append(y)

    X, X_len =  batch_to_ids(X_str, word2id, max_len)
    Y, Y_len =  batch_to_ids(Y_str, word2id, max_len)
    dataset = tf.data.Dataset.from_tensor_slices((X,Y)).shuffle(buffer_size)
    # dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset



class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):

        # x: batch_size, seq_len > 128, 1
        x = self.embedding(x)
        # x : (batch = 128 , em_dim = 20)
        output, state = self.gru(x,initial_state=hidden)
        # (128,512)
        # 输出的形状 == （批大小 * 1，隐藏层大小）
        output = tf.reshape(output, (-1, output.shape[2]))
        # 输出的形状 == （批大小，vocab）
        x = self.fc(output)
        return x, state


def train(EPOCHS = 10):
    # EPOCHS = 50
    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden, batch_size)  #
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # 每 2 个周期（epoch），保存（检查点）一次模型
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence):
    inputs,_ = sentence_to_ids(sentence, word2id, 10)
    inputs = tf.expand_dims(inputs,0)
    print('inpus raw:',repr(inputs))  #
    inputs = tf.convert_to_tensor(inputs) # (20,) => batch_size, seq_len
    print('to tensor:',repr(inputs))  #  tensor: <tf.Tensor: shape=(20,), dtype=int32,

    result = ''

    hidden = [tf.zeros((1, units))]
    # hidden = encoder.initialize_hidden_state()
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    # dec_input = tf.expand_dims([word2id['^']] * batch_size, 1) # on train_step
    dec_input = tf.expand_dims([word2id['^']], 0)
    print('dec_input', dec_input.shape)

    max_length_targ = 6
    for t in range(max_length_targ):
        predictions, dec_hidden = decoder(dec_input,dec_hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += id2word[predicted_id]

        if id2word[predicted_id] == '$':
            return result, sentence

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence

def translate(sentence):
    result, sentence = evaluate(sentence)
    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))


if __name__ == '__main__':
    # print(test_generate_equations())

    train_set, test_set = gen_data()


    # test batch to ids
    # sentences = train_set[0]
    # ids, sent_lens = batch_to_ids(sentences, word2id, max_len=10)
    # print('Input:', sentences)
    # print('Ids: {}\nSentences lengths: {}'.format(ids, sent_lens))
    # Input: ('1414-1725', '-311')
    # Ids: [[5, 8, 5, 8, 4, 5, 11, 6, 9, 2], [4, 7, 5, 5, 2, 0, 0, 0, 0, 0]]
    # Sentences
    # lengths: [10, 5]

    batch_size = 128
    embedding_dim = 20
    vocab_inp_size = len(word2id)
    units = 512
    max_len = 20

    # print("vocab_inp_size : {}".format(vocab_inp_size))
    # sample_input = [[5, 8, 5, 8, 4, 5, 11, 6, 9, 2], [4, 7, 5, 5, 2, 0, 0, 0, 0, 0]]
    #
    # sample_input = generate_batches(train_set,batch_size)
    # # (X_batch, Y_batch) = sample_input
    # X, X_seq_len = batch_to_ids(sample_input[0], word2id, max_len)
    # Y, Y_seq_len = batch_to_ids(Y_batch, word2id, max_len)

    dataset = generate_batch_dataset(train_set, batch_size, max_len)

    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
    # 样本输入
    sample_hidden = encoder.initialize_hidden_state()

    # example_input_batch, example_target_batch = next(iter(dataset))
    # print(example_input_batch.shape)  # tensor (128,10) dtype int32
    # print(example_target_batch.shape) # tensor (128, 6)
    #
    # sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    # print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))
    # output shape: (batch size, sequence length, units)(128, 10, 512)
    #  Hidden state shape: (batch size, units) (128, 512)

    # print(encoder.summary())

    decoder = Decoder(vocab_inp_size, embedding_dim, units, batch_size)
    # sample_decoder_output, _= decoder(tf.random.uniform((128, 1)),sample_hidden)
    # print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
    # Decoder output shape: (batch_size, vocab size)(768, 15)  should be (128, 15)
    # now is output shape: (batch_size, seq_steps, vocab size) (128, 6, 15)
    # now is (128, 15)
    # print(decoder.summary())


    optimizer = tf.keras.optimizers.Adam()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # checkpoint
    # checkpoint_dir = './ckpts/seq2seq_equiations'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    @tf.function
    def train_step(inp, targ, enc_hidden, batch_size):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([word2id['^']] * batch_size, 1)
            # 教师强制 - 将目标词作为下一个输入
            for t in range(1, targ.shape[1]):
                # 将编码器输出 （enc_output） 传送至解码器
                predictions, dec_hidden= decoder(dec_input, dec_hidden)
                loss += loss_function(targ[:, t], predictions)
                # 使用教师强制
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    # train
    steps_per_epoch = len(train_set) // batch_size
    train(100)

    # evaluate
    print('restore the model')
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("evaluate the model")
    translate("8561+677$#")
    translate("6684-7182$")

 There are many Pre-Trained Embeddings, which are developed by Google and which have been Open Sourced.
Some of them are Universal Sentence Encoder (USE), ELMO, BERT, etc.. and it is very easy to reuse them in your code.
Code to reuse the Pre-Trained Embedding, Universal Sentence Encoder is shown below:

  !pip install "tensorflow_hub>=0.6.0"
  !pip install "tensorflow>=2.0.0"

  import tensorflow as tf
  import tensorflow_hub as hub

  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
  embed = hub.KerasLayer(module_url)
  embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])
  print(embeddings.shape)  #(3,128)
  
  TensorFlow Hub 是一个针对可重复使用的机器学习模块的库
  https://www.tensorflow.org/hub
  
  X=tf.keras.layers.Embedding(input_dim=vocab_size,
                            output_dim=300,
                            input_length=Length_of_input_sequences,
                            embeddings_initializer=matrix_of_pretrained_weights
                            )(ur_inp)
  
  
  
  
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense

from autoencoder.utility import plot_model_result, load_mnist_dataset, plot_loss

class Encoder(tf.keras.layers.Layer):
    def __init__(self, n_dims, name='encoder', **kwargs):
        super(Encoder, self).__init__(name = name, **kwargs)
        self.n_dims = n_dims 
        self.n_layers = len(n_dims)

        self.hidden_layers = []
        for i in range(len(n_dims)-1):
            hiddent_layer = Dense(n_dims[i], activation=tf.nn.relu)
            self.hidden_layers.append(hiddent_layer)
        
        self.encode_layer = Dense(n_dims[-1], activation=tf.nn.relu)

    @tf.function
    def call(self, input):
        x = self.hidden_layers[0](input)
        for i in range (1,len(self.hidden_layers)):
            x = self.hidden_layer[i](x)
        
        return self.encode_layer(x)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, n_dims, name ='decoder', **kwargs): 
        super(Decoder, self).__init__(name = name, **kwargs) 
        self.n_dims = n_dims 
        self.n_layers = len(n_dims) 

        self.hidden_layers = []
        for i in range(len(n_dims)-1):
            hiddent_layer = Dense(n_dims[i], activation=tf.nn.relu)
            self.hidden_layers.append(hiddent_layer)

        self.recon_layer = Dense(n_dims[-1], activation ='sigmoid')
          
    @tf.function         
    def call(self, inputs): 
        x = self.hidden_layers[0](input)
        for i in range (1,len(self.hidden_layers)):
            x = self.hidden_layer[i](x)

        return self.recon_layer(x)

class FcAutoEncoder(tf.keras.Model):
    # def __build_input(self):
    #     # self.flatten_layer = tf.keras.layers.Flatten()
    #     pass 
    
    # def __compute_loss(self):
    #     self.loss = tf.keras.losses.MeanSquaredError() 
    #     # pass 
    
    # def __perform_optimization(self):
    #     # self.loss = 
    #     pass
    
    def __init__(self,enc_n_dims, dec_n_dims, name='fcautoencoder', **kwargs):
        super(FcAutoEncoder, self).__init__(name = name, **kwargs) 
        self.enc_n_dims = enc_n_dims
        self.dec_n_dims = dec_n_dims

        self.encoder = Encoder(enc_n_dims)
        self.decoder = Decoder(dec_n_dims)

        # define loss
        # self.__compute_loss()
        # self.op = 


    @tf.function
    def call(self,inputs):
        x = self.encoder(inputs)
        return self.decoder(x)
    
    #######################################
    # train
    def train_on_batch(self,train_data,test_data,num_epochs=10):
        # pred = self.call(input)
        # pass 
        self.compile(optimizer=tf.optimizers.Adam(0.01),
               loss = 'mse'
               )

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/' + 'train_name' + '/test_{epoch}',
                save_best_only=True,
                monitor='val_loss',
                verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir='log')
        ]
        self.history = self.fit(train_data,
                     validation_data=test_data,
                     epochs=num_epochs,
                     callbacks=callbacks)

    
    #######################################
    # predit
    # for predict only
    def predict_for_batch(self):
        pass 
    

    # for evaluation
    def predict_for_batch_with_loss(self):
        pass

if __name__ == '__main__':
    
    ae = FcAutoEncoder(enc_n_dims=[200, 32],dec_n_dims=[200,784])

    train_data, test_data, x_test = load_mnist_dataset(128)
    ae.train_on_batch(train_data, test_data)
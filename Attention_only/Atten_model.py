import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer,Embedding,GRU,Dense

class Encoder(Model):
    def __init__(self,vocab_size,embedding_dim,enc_units,batch_size):
        """
        Encoder object initialised
        :param vocab_size: input_vocab_size
        :param embedding_dim: input_embedding_dim output
        :param enc_units: output of GRU/LSTM layer
        :param batch_size:
        """
        # This is caled when Encoder object is initialised
        print("Encoder object initialised..")
        # Providing our Encoder with properies of super class
        super(Encoder,self).__init__()

        self.enc_units = enc_units
        self.batch_size = batch_size

        # Initialising all the layers needed in the Encoder Model
        self.embedding = Embedding(vocab_size,embedding_dim)
        self.gru = GRU(self.enc_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

    def call(self,x,hidden):
        """

        :param x: Input to the Encoder
        :param hidden: previous hidden state (s-1) of the decoder
        :return: output(hidden _states) along each time_step and final hidden_state only
        """
        # x is the input to the encoder
        # hidden is hidden_state (s-1) of the decoder
        print("X shape: {}".format(x.shape))
        x = self.embedding(x)   # Passing Input through the Embedding layer
        print("X shape after embedding: {}".format(x.shape))
        output,state = self.gru(x,initial_state = hidden)

        # The Gru will output output for each hidden_state
        # Along with last state
        # the hidden state (s-1) of the decoder is passed to the encoder via initial state
        return output,state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))



encoder = Encoder(vocab_size = 50, embedding_dim = 256, enc_units= 1024, batch_size = 64)

class BAttention(Layer):
    def __init__(self,units):
        print("BAttention Layer Initialised")
        # Giving it properties of the layer
        super(BAttention,self).__init__()

        # Initialising all the Params needed in the layer

        self.W1 = Dense(units)  # learnable weight params for input as --> query
        self.W2 = Dense(units)  # learnable weight params for input as --> values
        self.V = Dense(1)

        # Query ---- > Previous hidden state of the decoder(s-1)
        # Values --- > ALl the outputs(hidden states at each time_step) of the Encoder
        # query --> hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, decoder_hidden size)
        # values shape == (batch_size, input_max_len, encoder_hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score

    def call(self,query,values):
            """
            This is where attention wieghts will be calculated
            It takes query which is previous hidden state of the decoder
            and Values which are all the hidden states of the encoder
            :param self:
            :param query:
            :param values:
            :return:
            """
            query_with_time_axis = tf.expand_dims(query, 1)

            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            score = self.V(tf.nn.tanh(
                self.W1(query_with_time_axis) + self.W2(values)))
            # query_with_time_axis shape == (batch_size, 1, decoder_hidden size)
            # values shape == (batch_size, input_max_len, encoder_hidden size)

            # attention_weights shape == (batch_size, max_length, 1)
            attention_weights = tf.nn.softmax(score, axis=1)
            # values shape == (batch_size, input_max_len, encoder_hidden size)

            context_vector = attention_weights * values          # (batch_size,max_len,encoder_hidden_size)
            context_vector = tf.reduce_sum(context_vector, axis=1)  # context_vector shape after sum == (batch_size, hidden_size)


            return context_vector, attention_weights


class Decoder(Model):
    def __init__(self,out_vocab_size,embedding_dim,dec_units,batch_size):
        print("Decoder object Initialised")

        # Giving it all the powers of the Super class
        super(Decoder,self).__init__()
        self.dec_units = dec_units
        self.batch_size = batch_size

        # Initalise all the layers needed in the Decoder
        self.embedding = Embedding(out_vocab_size,embedding_dim)
        self.gru = GRU(dec_units,
                       return_sequences=True,
                       return_state=True,
                       recurrent_initializer='glorot_uniform')

        self.fc = Dense(out_vocab_size)  # To make predictions
        self.attention = BAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
            # x here is single word/character
            # Calculating the Context vector and attention weights
            # enc_output shape == (batch_size, max_length, hidden_size)
            # context_vector_shape ==(batch_size, hidden_size)
            context_vector, attention_weights = self.attention(hidden, enc_output)

            # X shape before passing through embedding ==(batch_size,1)

            # x shape after passing through embedding == (batch_size, 1, embedding_dim)
            x = self.embedding(x)

            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            # context vector before exapnd_dims == (batch_size,hidden_size)
            # context vector after expand_dims == (batch_size,1,hidden_size)

            x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

            # Input to GRU --> (batch_size,1,embedding_dim + hidden_size)
            # Output from GRU --> (batch_size,1,hidden_size)
            # passing the concatenated vector to the GRU
            output, state = self.gru(x)

            # output shape == (batch_size * 1, hidden_size)
            output = tf.reshape(output, (-1, output.shape[2]))

            # output shape == (batch_size, vocab)
            # Making Prediction
            x = self.fc(output)
            print(x.shape)

            return x, state, attention_weights


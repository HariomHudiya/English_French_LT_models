import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding,LayerNormalization


# Input is of shape --> (batch_size,seq_len,d_model)
# Which basically boils down that we have a batch contains many sequences(batch_size)
# and each sequence is of (seq_len)
# with each charcter/word in the seq_len is of d_model dimension

# Goal we need to encode (batch_of_sequences each of which is seq_len long and each word in seq is encoded in d_model dim) such that
# Each position is relatively encoded

# Lets take pos = 50
# d_model = 256
def get_angles(pos, i, d_model):
    """

    :param pos: Total positions in the seq  (seq_len)  50
    :param i: index of the word
    :param d_model: encoding of the word  256
    :return: angle a vector (position,1) number of angles
    """
    # position will be a vector of (pos,1) dimension -- >  Every position is unique
    # d_model will be a vector of (1,d_model) dimension -- > 1 position will have d_model size vector

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    It'll apply sin_fn() to even indexs of the angle
    and cos_fn() to odd indexs of the angle,
    where angle is vector of d_model dimension and we have position number
    of angles...
    :param position:  seq_len
    :param d_model: encoding of the word
    :return:
    """
    print("Creating Positional Encoding.....")
    print("Position Dimensions: ")
    position = np.arange(position)  # Creating a vector [0,1,2,..position(50)] # Index generated --> (50,)
    print("Before expanding dimension: {}".format(position.shape))
    position = np.expand_dims(position, axis=-1)  # Expanding generated vector --> (50,1)
    print("After Expanding Dimension: {}".format(position.shape))
    print("------------------------")
    print("Angle Dimensions: ")

    index = np.arange(d_model)  # Creating a vector [0,1,2,...d_model(256)] --> (256,)
    print("Before expanding dimension: {}".format(index.shape))
    index = index[np.newaxis, :]  # Unique index for each d_model
    print("After expanding dimension: {}".format(index.shape))

    angle_rads = get_angles(position,
                            index,
                            d_model)
    print("Angle rads shape {}: ".format(angle_rads.shape))

    # Uses Broadcasting ...

    # apply sin to even indices in the array; 2i
    # For all 50 position
    # in every single position of d_model(256 dimension)
    # Consider even indexes in sin case
    # odd indexes in cos case
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # Adding batch dimesion to postional encoding so its of shape (batch_size, seq_len , d_model)
    pos_encoding = np.expand_dims(angle_rads, axis=0)
    # print(pos_encoding.shape)

    return tf.cast(pos_encoding, dtype=tf.float32)  # tf.cast converts the d-type
    # Casts a tensor to a new type.




def create_padding_mask(seq):
    """
    Returns a pad vector
    a pad vector contains 1 if something is present in the seq else 0
    :param seq: a vector with some numbers
    :return: pad vector
    """
    print("Creating Padding Mask......")
    # Can also work on batch_of_sequences as (batch_size,seq_len)

    pad_seq = tf.math.equal(seq, 0)  # Returns truth value of x==y
    # Each value in the 50_dim sequence will be compared to 0 and return boolean
    seq = tf.cast(pad_seq, tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)



print("-----------------")


def create_look_ahead_mask(size):
    """

    :param size:
    :return: a upper triangular matrix  (square matrix of dimension (size,size))
    """
    print("Creating look_ahead_mask...............")
    # Converting tf.ones to lower triangular matrix
    # Subtracting by 1 (which will be broadcasted to convert it into upper triangular matrix)
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1,
                                   0)  # num_lower is sub_diagonals to keep --> all(negative) ,
    # num_upper is super_diagonals to keep --> 0 (num)
    return mask  # (seq_len, seq_len)




def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculates attention_weights(score) and attention_output
    :param q:Query -->  query shape == (..., seq_len_q, depth)
    :param k:key --> key shape == (..., seq_len_k, depth)
    :param v:Value --> value shape == (..., seq_len_v, depth_v)
    :param mask: Mask vector (so that softmax of high mask becomes 0)loat tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
    :return:
    """
    print("Calculating Scaled dot product attention")
    print("Query shape: {}".format(q.shape))
    print("Key shape: {}".format(k.shape))
    print("Value shape: {}".format(v.shape))

    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # np.dot(Q,K.T)
    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # Adding mask if present
    if mask is not None:
        scaled_attention_logits += (
                    mask * -1e9)  # Multiplying mask with big value so that softmax drives to 0 at that pos

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # We need to find values using keys as the probability distribution for a specific q
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    # attention_weights are nothing but attention scores (probablitiy distribution over key)

    output = tf.matmul(attention_weights, v)  ## (...,seq_len_q , depth_v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        """
        Initialises the class object called when instance is created
        :param d_model:
        :param num_heads:
        """
        print("Instantiating Multhead Attention object..")
        # Initialising super class to access it's benefits
        # Basically we'll call their method and delegate work to super class function as and when needed
        super(MultiHeadAttention, self).__init__()

        # Making attributes from passed in parameters
        self.num_heads = num_heads
        self.d_model = d_model
        print("D_model: {}".format(d_model))
        print("Num_heads: {}".format(num_heads))

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        # We will assert that d_model is divisble by num_heads
        # Because we will spit d_model into num_heads so spliting should be possible and even

        # Initialising the Weight matrices of q,k and v
        # Which will be learnable parameters and will be updated
        # So we will put them in the form of Dense Layer

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        # Initialising the layer needed
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        This is basically a helper function which will split the
        d_model into --> (num_heads,depth) where num_heads * depth = d_model

        We will need the result back in (batch-size,num_heads,seq_len,depth)

        This basically says for a batch of sequences we will pass it num_heads
        where each sequence is of len seq_len and and each word is of depth (dimension)

        :param x: input of shape (batch_size,seq_len,d_model)
        :param batch_size:
        :return:
        """

        x = tf.reshape(x, (batch_size, -1, self.num_heads,
                           self.depth))  # -1 indicates --> flattening of vector along that axis (seq_len in this case)
        return tf.transpose(x, perm=[0, 2, 1, 3])  # Re-ordering of axis

    def call(self, v, k, q, mask):
        """
        This is the code that will run when we call and instance of the class MuliheadAttention
        :param v: value shape == (..., seq_len_v, depth_v)
        :param k: key shape == (..., seq_len_k, depth)
        :param q: query shape == (..., seq_len_q, depth)
        :param mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
        :return:
        """
        print("Calculating Multihead Attention...")
        batch_size = np.shape(q)[0]
        print("Initial query,key and value")

        print("Query shape (batch_size,seq_len,d_model): {}".format(q.shape))
        print("Key shape (batch_size,seq_len,d_model): {}".format(k.shape))
        print("Value shape (batch_size,seq_len,d_model): {}".format(v.shape))
        print("---------------------")
        # Calling the layers initialised above

        q = self.wq(q)      # (batch_sie,seq_len,d_model)
        # We are basically passing q as input to the hidden layer(wq)
        # Similarly

        k = self.wk(k)
        v = self.wv(v)

        # With the above operation we have each of q,k and v of dimension # (batch_sie,seq_len,d_model)
        print("Batch Size: {}".format(batch_size))

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)  # Transpose
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)  # seq_len_k == seq_len_v
        print("After Split Query,key and Value shapes")
        print("Query shape (batch_size, num_heads, seq_len_q, depth): {}".format(q.shape))
        print("Key shape (batch_size, num_heads, seq_len_q, depth): {}".format(k.shape))
        print("Value shape (batch_size, num_heads, seq_len_q, depth): {}".format(v.shape))

        # Passing the calculated q,k and value for attention weights and attention output cal

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        print("scaled attention transpose weight: (batch_size, seq_len_q, num_heads, depth) {}".format(
            scaled_attention.shape))

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        print("scaled attention concat shape: (batch_size, seq_len_q, d_model) {}".format(concat_attention.shape))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """

    :param d_model:
    :param dff:
    :return:
    """
    print("Executing point_wise_feed_forward_network")
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class EncoderLayer(Layer):
    def __init__(self,d_model,num_heads,dff,rate = 0.1):
        """
        This is called when object of the class is instantiated
        """
        print("******************************************************")

        # Assigning all the powers of the super layer class
        super(EncoderLayer,self).__init__()
        print("Instantiating Encoder layer")
        # We will pass all the parameters needed by the layers in the class
        # as parameter to the class
        # and some parameters will be hard_coded within the function call (see epsilon_value)

        ## Instantiating the Layers we will be needing

        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ffn = point_wise_feed_forward_network(d_model,dff)

        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

        # We will apply dropout between every layer
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self,x,training,mask):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)  # x can be either of q,k,v
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm_1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm_2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2








class DecoderLayer(Layer):
    def __init__(self,d_model,num_heads,dff,rate = 0.1):
        # This layer is called when object of Decoder Layer is initialised
        print("************************************")
        print("Decoder Layer object initialised")
        # Accessing all the super class properties by initialising
        super(DecoderLayer,self).__init__()

        # Initialising all the layers needed
        self.mha1 = MultiHeadAttention(d_model,num_heads)
        self.mha2 = MultiHeadAttention(d_model,num_heads)
        self.ffn = point_wise_feed_forward_network(d_model,dff)

        self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = LayerNormalization(epsilon=1e-6)

        self.dropout_1 = Dropout(rate)
        self.dropout_2 = Dropout(rate)
        self.dropout_3 = Dropout(rate)

    def call(self,x,enc_output,training,look_ahead_mask,padding_mask):
        """
        This is called when object of this class is called
        :param x: Input vector (batch_size,seq_len,d_model)
        :param enc_output: (batch_size,seq_len,d_model)
        :param training: keyword argument for Dropout
        :param look_ahead_mask: for Masked Multihead Attention
        :param padding_mask: for MHA comming from Encoder
        :return:
        """

        attn_output_1,attn_weights_block_1 = self.mha1(x,x,x,look_ahead_mask)
        attn_output_1 = self.dropout_1(attn_output_1,training=training)
        out1 = self.layer_norm_1(x + attn_output_1)

        attn_output_2,attn_weights_block_2 = self.mha2(enc_output,enc_output,out1,padding_mask)
        attn_output_2 = self.dropout_2(attn_output_2,training=training)
        out2 = self.layer_norm_2(out1 + attn_output_2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout_3(ffn_output)
        out3 = self.layer_norm_3(out2 + ffn_output)

        return out3, attn_weights_block_1 , attn_weights_block_2



class Encoder(Layer):
    def __init__(self,input_vocab,d_model,num_layers,maximum_position_encoding,num_heads,dff,rate =0.1):
        """
        This will be called when encoder object is initialised
        All the parameters needed by the Layers inside the Enocder will be passed during initialisation
        :param input_vocab: Size of input_vocab (needed by Encoding layer)
        :param d_model: Size of Feature embedding (needed by all the layers)
        :param num_layer: Number of Encoder Layer Needed
        :param maximum_position_encoding:
        :param num_heads: Number of Multiheads needed (ensure that d_model is divisible by num_heads)
        :param dff: Number of units Needed by Feed_forward layer in the encoder layer
        :param rate: Dropout rate for training only
        """
        # Accessing all the features of the base class
        print("++++++++++++++++++++++++++++++++++++++")
        print("Encoder object initialised")
        super(Encoder,self).__init__()

        self.d_model = d_model
        self.num_layer = num_layers

        # Instantiating all the layers needed
        self.embedding = Embedding(input_vocab +1 ,d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]

        # List of Encoder blocks/layers

        self.dropout = Dropout(rate)

    def call(self,x,training,mask):

        seq_len = x.shape[1]
        # Adding emmbedding and Positional embedding

        x = self.embedding(x)  # Output (batch_size,input_seq_len,d_model)
        x += self.pos_encoding[:,:seq_len,:]

        # For all the batches for all the fetures only upto seq_len

        x =self.dropout(x,training = training)

        for i in range(self.num_layer):
            x = self.enc_layers[i](x,training,mask)

        return x #(batch_size,input_seq_len,d_model)




class Decoder(Layer):
    def __init__(self,target_vocab_size,d_model,maximum_positional_encoding,num_heads,dff,num_layers,rate=0.1):
        """
        This is called when Decoder object is initialised
        :param target_vocab_size: Size of the target_vocab (input needed by Embedding Layer)
        :param d_model: number of features needed by all layers
        :param maximum_positional_encoding:
        :param num_heads: Number of heads to be split
        :param dff: Number of units (needed by feed forward netword)
        :param num_layers: Number of layers
        :param rate: Dropout rate
        """
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("Decoder object initialised")

        # Accessing Features of the base class
        super(Decoder,self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Instantiating all the layers needed in the decoder layer

        self.embedding = Embedding(target_vocab_size +1 , d_model)
        self.pos_encoding = positional_encoding(maximum_positional_encoding,d_model)

        self.dec_layers = [DecoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]

        self.dropout = Dropout(rate)

    def call(self,x,encoder_output,training,look_ahead_mask,padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights= {}

        x = self.embedding(x) # (batch_size,target_seq_len,d_model)
        x += self.pos_encoding[:,:seq_len,:]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, encoder_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights





class Transformer_model(Model):
    def __init__(self,num_layers,d_model,input_vocab,target_vocab_size,pe_input,pe_target,num_heads,dff,rate = 0.1):

        # Accessing all the features of super class
        print("++++++++++++++++++++++++++++++")
        print("Transformer object instantiated")
        super(Transformer_model,self).__init__()

        # Instantiating the blocks needed
        self.encoder = Encoder(input_vocab,d_model,num_layers,pe_input,num_heads,dff,rate)

        self.decoder = Decoder(target_vocab_size,d_model,pe_target,num_heads,dff,num_layers,rate)

        self.final_layer = Dense(target_vocab_size + 1)
        print("Transformer Object Initialised Successfully..............")

    def call(self,inp,tar,training,enc_padding_mask,look_ahead_mask,dec_padding_mask):

        print("Call to Transformer executed..........")

        enc_output = self.encoder(inp,training,enc_padding_mask) # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights    # dec_output.shape == (batch_size, tar_seq_len, d_model)




if __name__ == "__main__":
    pos_encoding = positional_encoding(50, 512)
    print (pos_encoding.shape)

    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
    print("-----------------------------")
    x = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    print("Input Shape: {}".format(x.shape))
    print(create_padding_mask(x))

    x = tf.random.uniform((1, 3))
    temp = create_look_ahead_mask(x.shape[1])
    print(temp)
    print("--------------------------------")

    def print_out(q, k, v):
        temp_out, temp_attn = scaled_dot_product_attention(
            q, k, v, None)
        print('Attention weights are:')
        print(temp_attn)
        print('Output is:')
        print(temp_out)
        print("-----------------------")


    def scaled_dot_product_attention_checker():
        temp_k = tf.constant([[10, 0, 0],
                              [0, 10, 0],
                              [0, 0, 10],
                              [0, 0, 10]], dtype=tf.float32)  # (4, 3)  # (1,4,3)

        temp_v = tf.constant([[1, 0],
                              [10, 0],
                              [100, 5],
                              [1000, 6]], dtype=tf.float32)  # (4, 2)  # (1,4,2)
        # This `query` aligns with the second `key`,
        # so the second `value` is returned.
        temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
        print_out(temp_q, temp_k, temp_v)
        # This query aligns with a repeated key (third and fourth),
        # so all associated values get averaged.
        temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
        print_out(temp_q, temp_k, temp_v)
        # This query aligns equally with the first and second key,
        # so their values get averaged.
        temp_q = tf.constant([[10, 10, 0]], dtype=tf.float32)  # (1, 3)
        print_out(temp_q, temp_k, temp_v)
        temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)  # (1,3,3)
        print_out(temp_q, temp_k, temp_v)

    scaled_dot_product_attention_checker()

    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(y, k=y, q=y, mask=None)
    print("--------------------------------------------------------")
    print("Output Shape: {}".format(out.shape))
    print("Attention Shape: {}".format(attn.shape))
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 64))).shape)

    sample_encoder_layer = EncoderLayer(512, 8, 2048)

    sample_encoder_layer_output = sample_encoder_layer(
        tf.random.uniform((64, 43, 512)), False, None)

    print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)


    sample_decoder_layer = DecoderLayer(512, 8, 2048)

    sample_decoder_layer_output, _, _ = sample_decoder_layer(
        tf.random.uniform((64, 50, 512)), sample_encoder_layer_output,
        False, None, None)

    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)

    sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, input_vocab=8500,
                             maximum_position_encoding=10000)
    temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)

    sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)

    print (sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
    sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8,
                             dff=2048, target_vocab_size=8000,
                             maximum_positional_encoding=5000)
    temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)

    output, attn = sample_decoder(temp_input,
                                  encoder_output=sample_encoder_output,
                                  training=False,
                                  look_ahead_mask=None,
                                  padding_mask=None)

    print(output.shape, attn['decoder_layer2_block2'].shape)

    sample_transformer = Transformer_model(
        num_layers=2, d_model=512, num_heads=8, dff=2048,
        input_vocab=8500, target_vocab_size=8000,
        pe_input=10000, pe_target=6000)

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)

    fn_out, _ = sample_transformer(temp_input, temp_target, training=False,
                                   enc_padding_mask=None,
                                   look_ahead_mask=None,
                                   dec_padding_mask=None)

    print(fn_out.shape)  # (batch_size, tar_seq_len, target_vocab_size)





















import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Attn_utils import pkl_file_loader,pkl_file_saver
from tensorflow.keras.utils import to_categorical
import time
from Atten_model import Encoder,Decoder,BAttention
import os


max_len = 60
word_idx_english = pkl_file_loader("..\mappings\.word_idx_english.pkl")
word_idx_french = pkl_file_loader("..\mappings\.word_idx_french.pkl")
idx_word_english = pkl_file_loader("..\mappings\.idx_word_english.pkl")
idx_word_english = pkl_file_loader("..\mappings\.idx_word_english.pkl")
idx_word_french = pkl_file_loader("..\mappings\.idx_word_french.pkl")

eng_vocab_size = len(list(word_idx_english.keys())) + 2
french_vocab_size = len(list(word_idx_french.keys())) + 2

doc_path_english = "..\processed_data\.\\token_eng_file.txt"
doc_path_french = "..\processed_data\.\\token_fre_file.txt"

def one_hot(ds,vocab_size):
    return to_categorical(ds, num_classes=vocab_size)

def doc_loader_3_english(doc_path=doc_path_english ,num_samples=1280):
    """
    Loads a txt file or other data supported files
    :param doc_path: file path
    :return: contents of the file
    """
    print("Loading Doc.....")
    row_cnt = 0
    with open(doc_path,'r', encoding='UTF-8') as f:
       #content = f.read()    # Loads the whole FIle ## CAUTION :- May result in memory overload , solution dataset obj/ generator
       for row in f:
        row_cnt += 1
           #print(row_cnt)
        if num_samples != None:
            if row_cnt <= num_samples:
                temp_row = [int(i) for i in row.split()]
                if len(temp_row) < max_len:
                    temp_row = temp_row + [0] * (max_len - len(temp_row))
                    yield (temp_row)
                else:
                    temp_row = temp_row[:max_len+1]
                    yield (temp_row)
            else:
                break
        else:
            temp_row = [int(i) for i in row.split()]
            if len(temp_row) < max_len:
                temp_row = temp_row + ([0] * (max_len - len(temp_row)))
                yield (temp_row)
            else:
                temp_row = temp_row[:max_len + 1]
                yield (temp_row)


english_ds = tf.data.Dataset.from_generator(doc_loader_3_english,output_types=(tf.int32), output_shapes=None,
                                           args= None)




print(english_ds.element_spec)
for element in english_ds.take(5).as_numpy_iterator():
    print(element.shape)

print("English dataset Built Successfully")


def doc_loader_3_french(doc_path=doc_path_french ,num_samples=1280):
    """
    Loads a txt file or other data supported files
    :param doc_path: file path
    :return: contents of the file
    """
    print("Loading Doc.....")
    row_cnt = 0
    with open(doc_path,'r', encoding='UTF-8') as f:
       #content = f.read()    # Loads the whole FIle ## CAUTION :- May result in memory overload , solution dataset obj/ generator
       for row in f:
        row_cnt += 1
           #print(row_cnt)
        if num_samples != None:
            if row_cnt <= num_samples:
                row = row.strip()
                temp_row = [int(i) for i in row.split()]
                if len(temp_row) < max_len:
                    temp_row = temp_row + ([0] * (max_len - len(temp_row)))
                    #temp_row = one_hot(temp_row, french_vocab_size)

                    yield (temp_row)
                else:
                    temp_row = temp_row[:max_len + 1]
                    #temp_row = one_hot(temp_row, french_vocab_size)

                    yield (temp_row)

            else:
                break
        else:
            row = row.strip()
            temp_row = [int(i) for i in row.split()]
            if len(temp_row) < max_len:
                temp_row = temp_row + ([0] * (max_len - len(temp_row)))
                #temp_row = one_hot(temp_row, french_vocab_size)

                yield (temp_row)
            else:
                temp_row = temp_row[:max_len + 1]
                #temp_row = one_hot(temp_row, french_vocab_size)

                yield (temp_row)


french_ds = tf.data.Dataset.from_generator(doc_loader_3_french,output_types=(tf.int32), output_shapes=None,
                                           args= None)



print("French dataset Built Successfully")

ds = tf.data.Dataset.zip((english_ds,french_ds))
print("Zipped dataset built successfully")
#ds = ds.padded_batch(64,padded_shapes=((),()))
print(ds.element_spec)
print("Ds built Successfully")
ds = ds.batch(64)
for element_batch in ds.take(1).as_numpy_iterator():
    for element in element_batch:
        print(np.array(element).shape)


print("Building The model")
batch_size = 64
steps_per_epoch = 1280// batch_size
embedding_dim = 256
units = 1024

encoder = Encoder(eng_vocab_size,embedding_dim,enc_units=units,batch_size = batch_size)
decoder = Decoder(french_vocab_size,embedding_dim,dec_units=units,batch_size=batch_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  print(real.shape)
  mask = tf.math.logical_not(tf.math.equal(real[0], 0))
  print(mask)
  loss_ = loss_object(real, pred)
  print("Mask Shape:{}".format(mask.shape))
  print("Loss shape: {}".format(loss_.shape))
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  print(loss_)
  print("loss computes successfully")

  return tf.reduce_mean(loss_)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)



@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([word_idx_french['<start>']] * batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      print("Target Value: {}".format(targ[:,t]))
      print("Predictions Value: {}".format(predictions))
      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss


EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(ds.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 10 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

 

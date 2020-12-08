import tensorflow as tf
import numpy as np
from Transformer_implementaion.Transformer_utils import pkl_file_saver,pkl_file_loader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Transformer_implementaion.Transformer import Transformer_model,create_look_ahead_mask,create_padding_mask
from Transformer_implementaion.Transformer_scheduler import CustomSchedule
import time

max_len = 60
word_idx_english = pkl_file_loader(".word_idx_english.pkl")
word_idx_french = pkl_file_loader(".word_idx_french.pkl")
idx_word_english = pkl_file_loader(".idx_word_english.pkl")
idx_word_english = pkl_file_loader(".idx_word_english.pkl")
idx_word_french = pkl_file_loader(".idx_word_french.pkl")

eng_vocab_size = len(list(word_idx_english.keys())) + 2
french_vocab_size = len(list(word_idx_french.keys())) + 2

doc_path_english = ".\\token_eng_file.txt"
doc_path_french = ".\\token_fre_file.txt"

def doc_loader_3_english(doc_path=doc_path_english ,num_samples=20000):
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

for element in english_ds.take(5).as_numpy_iterator():
    print(element)

print("English dataset Built Successfully")


def doc_loader_3_french(doc_path=doc_path_french ,num_samples=None):
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
                    yield (temp_row)
                else:
                    temp_row = temp_row[:max_len + 1]
                    yield (temp_row)

            else:
                break
        else:
            row = row.strip()
            temp_row = [int(i) for i in row.split()]
            if len(temp_row) < max_len:
                temp_row = temp_row + ([0] * (max_len - len(temp_row)))
                yield (temp_row)
            else:
                temp_row = temp_row[:max_len + 1]
                yield (temp_row)


french_ds = tf.data.Dataset.from_generator(doc_loader_3_french,output_types=(tf.int32), output_shapes=None,
                                           args= None)


print("French dataset Built Successfully")

ds = tf.data.Dataset.zip((english_ds,french_ds))
print("Zipped dataset built successfully")
for eng_element,fre_element in ds.take(5).as_numpy_iterator():
    print("Token English Sentence: {}".format(eng_element))
    temp_eng_sent = []
    for token in eng_element:
        if token != 0:
            if token in idx_word_english.keys():
                temp_eng_sent.append(idx_word_english[token])
            else:
                temp_eng_sent.append(word_idx_english["<UNK>"])

    print("Actual Sentence: {}".format(temp_eng_sent))
    print("Token Fench Sentence: {}".format(fre_element))
    temp_fre_sent = []
    for token in fre_element:
        if token != 0:
            if token in idx_word_french.keys():
                temp_fre_sent.append(idx_word_french[token])
            else:
                temp_fre_sent.append(word_idx_french["<UNK>"])

    print("Actual Sentence: {}".format(temp_fre_sent))
    print("------------------------------------------------")


#ds = ds.padded_batch(64,padded_shapes=((),()))
print(ds.element_spec)
print("Ds built Successfully")
ds = ds.batch(64)
for element_batch in ds.take(1).as_numpy_iterator():
    for element in element_batch:
        print(np.array(element).shape)

d_model = 256
print("Running Transformer............")

transformer = Transformer_model(num_layers=8,d_model=256,input_vocab=eng_vocab_size,
                                target_vocab_size=french_vocab_size,pe_input=eng_vocab_size,
                                pe_target=french_vocab_size,num_heads=8,dff=512)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# Since the target sequences are padded, it is important to apply a padding mask when calculating the loss.


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

#
def accuracy_function(real, pred):
    accuracies = np.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

checkpoint_path = "./checkpoints/train_transformer"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

EPOCHS = 20

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
# ]
#
#
# @tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tf.cast(tar_real,tf.int32), tf.cast(predictions,tf.int32)))


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(ds):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("Epoch Number: {} , Batch Number: {}, Input Shape: {} , Target Shape: {}".format(epoch,batch,inp.shape,tar.shape))
        train_step(inp, tar)

        if batch % 50 == 0:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))



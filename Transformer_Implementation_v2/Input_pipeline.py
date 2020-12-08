from Transformer_implementaion.Transformer_utils import pkl_file_loader,pkl_file_saver
import tensorflow as tf
import numpy as np

doc_path_english = ".\clean_eng_file.txt"
doc_path_french = ".\clean_fre_file.txt"

def doc_loader_2_english(doc_path=doc_path_english ,num_samples=None):
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
                yield (row)
            else:
                break
        else:
            yield(row)

english_ds = tf.data.Dataset.from_generator(doc_loader_2_english,output_types=(tf.string), output_shapes=None,
                                           args= None)


print("English dataset Built Successfully")


def doc_loader_2_french(doc_path=doc_path_french ,num_samples=None):
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
                yield (row)
            else:
                break
        else:
            yield(row)

french_ds = tf.data.Dataset.from_generator(doc_loader_2_french,output_types=(tf.string), output_shapes=None,
                                           args= None)


print("French dataset Built Successfully")

ds = tf.data.Dataset.zip((english_ds,french_ds))
print("Zipped dataset built successfully")
for element in ds.take(5).as_numpy_iterator():
    print(element)
# Batching Dataset
ds = ds.batch(64)

word_counter_save_path_english = '.word_counter_english.pkl'
word_counter_save_path_french = '.word_counter_french.pkl'

word_count_english = pkl_file_loader(word_counter_save_path_english)
word_count_french = pkl_file_loader(word_counter_save_path_french)

print("Number of English Tokens: ")
print(len(word_count_english.keys()))
print("Number of French Tokens: ")
print(len(word_count_french.keys()))

# Sorting so that word_index order become consistent

english_tokens = sorted(list(word_count_english.keys()))
french_tokens = sorted(list(word_count_french.keys()))


def create_mappings(tokens, debuggining_info=True):
    """

    :param tokens: sorted tokens list
    :return: word_to_idx and idx_to_word
    """
    word_idx = {}
    for i, word in enumerate(tokens, start=2):  # Leave 0 for padding and 1 for unkown tokens
        word_idx[word] = i
    word_idx["<UNK>"] = 1

    idx_word = {}
    for i, word in enumerate(tokens, start=2):
        idx_word[i] = word
    idx_word[1] = "<UNK>"

    if debuggining_info:
        print("Length of Word index: ", len(word_idx.items()))  # num_words + 1 (Unknown Token / 0 padding)
        print("Length of index word mapping: ", len(idx_word.items()))  # num_words + 1 (UNK TOken/ (0--padding)
        print("Length of tokens: ", len(tokens))  # num_words
        #print("length of updated word counter : ", len(updated_word_count.items()))  # num_words

    return word_idx, idx_word

word_idx_english , idx_word_english = create_mappings(english_tokens)
word_idx_french , idx_word_french = create_mappings(french_tokens)

word_idx_english_path = ".word_idx_english.pkl"
pkl_file_saver(word_idx_english_path,word_idx_english)


idx_word_english_path = ".idx_word_english.pkl"
pkl_file_saver(idx_word_english_path,idx_word_english)


word_idx_french_path = ".word_idx_french.pkl"
pkl_file_saver(word_idx_french_path,word_idx_french)


idx_word_french_path = ".idx_word_french.pkl"
pkl_file_saver(idx_word_french_path,idx_word_french)


# start and end token already created

vocab_size_english = len(english_tokens) + 2  # (num_words + 1 (unknown token) + 1 (0 padding to make all equal length)

vocab_size_french = len(french_tokens) + 2  # (num_words + 1 (unknown token) + 1 (0 padding to make all equal length)
print("Vocab size English : %s"%vocab_size_english)
print("Vocab size French : %s" %(vocab_size_french))

print("First 20 English Tokens: ")
print(english_tokens[:20])
print("------------------------------")
print("First 20 French Tokens: ")
print(french_tokens[:20])
print("English Word Mappings: ")
print(word_idx_english["<start>"])
print(word_idx_english["<end>"])
print(word_idx_english["<UNK>"])
print("---------------------------------------------------------")
print("French word index mapping")
print(word_idx_french["<start>"])
print(word_idx_french["<end>"])
print(word_idx_french["<UNK>"])
#

## Note --> start token and end token positions


print("========================Encoding the Text: =================")
counter = 0
english_counter = 0
german_counter = 0

english_file_path = "token_eng_file.txt"
french_file_path = "token_fre_file.txt"

def dataset_saver_eng(sent):
    with open(english_file_path,'a',encoding='UTF-8') as f:
        f.write(sent)
        f.write("\n")
        print("Token Sentence Successfully written")

def dataset_saver_french(sent):
    with open(french_file_path,'a',encoding='UTF-8') as f:
        f.write(sent)
        f.write("\n")
        print("Token Sentence Successfully Written")


batch_counter = 0
translation_counter_eng = 0
translation_counter_fre = 0

for element in ds.as_numpy_iterator():     # Batch File # 64 Translations
    english_trans_batch, french_trans_batch = np.array_split(element, 2)  # file # 1 Translation
    #print(english_trans)
    batch_counter += 1
    print("Batch Accesssed...{}".format(batch_counter))
    print("=======================Working On English sentences batch====================")
    for english_trans in english_trans_batch:
        print("Single Batch Acccessed")
        for english_sent in english_trans:
            temp_eng_sent = []
            english_sent = english_sent.decode("UTF-8")
            translation_counter_eng += 1
            print("English Sentence Accessed..")
            for word in english_sent.split():
                if word not in word_count_english:
                        temp_eng_sent.append(str(word_idx_english["<UNK>"]))
                temp_eng_sent.append(str(word_idx_english[word]))
            dataset_saver_eng(" ".join(temp_eng_sent))
    print("=======================Working On French sentences batch====================")

    for french_trans in french_trans_batch:
        print("Single French  Batch Acccessed")
        for french_sent in french_trans:
            temp_french_sent = []
            print("French Sentence Accessed")
            french_sent = french_sent.decode("UTF-8")
            translation_counter_fre += 1
            for word in french_sent.split():
                if word not in word_count_french:
                    temp_french_sent.append(str(word_idx_french["<UNK>"]))
                temp_french_sent.append(str(word_idx_french[word]))
            dataset_saver_french(" ".join(temp_french_sent))

    print("====================BATCH PROCESSED SUCCESSFULLY=============================")

print("*************************Summary*********************************")
print("Batches processed: {}".format(batch_counter))
print("English Sentences Processed: {}".format((translation_counter_eng)))
print("French Sentences Processed: {}".format((translation_counter_fre)))
print("------------------------------------------------------------------")



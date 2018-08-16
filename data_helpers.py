import jieba_fast as jieba.analyse
import codecs
import numpy as np
from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def wrapper():
        stime = time.clock()
        func()
        endtime = time.clock()
        print("Runtime is {}".format(endtime-stime))


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(codecs.open(positive_data_file, "r", "utf-8").readlines())
    positive_examples = [
        [item for item in jieba.analyse.extract_tags(s, withWeight=False, topK=20, allowPOS=('n', 'v', 'nt', 'vn'))] for
        s in positive_examples]
    negative_examples = list(codecs.open(negative_data_file, "r", "utf-8").readlines())
    negative_examples = [
        [item for item in jieba.analyse.extract_tags(s, withWeight=False, topK=20, allowPOS=('n', 'v', 'nt', 'vn'))] for
        s in negative_examples]

    # Combine lists
    x_text = positive_examples + negative_examples

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_word2vector(filename):

    """
    Word2Vector_vocab – list of the words that we now have embeddings for
    Word2Vector_embed – list of lists containing the embedding vectors
    embedding_dict – dictionary where the words are the keys and the embeddings are the values
    """

    Word2Vector_vocab = []
    Word2Vector_embed = []
    embedding_dict = {}

    with codecs.open(filename, 'r', 'utf-8') as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab_word = row[0]
            Word2Vector_vocab.append(vocab_word)
            embed_vector = [float(i) for i in row[1:]]  # convert to list of float
            embedding_dict[vocab_word] = embed_vector
            Word2Vector_embed.append(embed_vector)

        print('Word2Vector Loaded Successfully')
        return Word2Vector_vocab, Word2Vector_embed, embedding_dict


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
@timeit           
def load_data(data_file):
    """
    Loads data from files, extracts tag words
    """
    # Load data from files
    data = list(codecs.open(data_file, "r", "utf-8").readlines())
    x_text = [
        [item for item in jieba.analyse.extract_tags(s, withWeight=False, topK=20, allowPOS=('n', 'v', 'nt', 'vn'))] for
        s in data]

    return x_text


       

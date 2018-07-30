import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import codecs
import os
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def load_word2vector(filename):

    """
    Word2Vector_vocab – list of the words that we now have embeddings for
    Word2Vector_embed – list of lists containing the embedding vectors
    embedding_dict – dictionary where the words are the keys and the embeddings are the values
    """

    Word2Vector_vocab = []
    Word2Vector_embed = []
    embedding_dict = {}

    with codecs.open(filename, 'r',"utf-8") as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            vocab_word = row[0]
            Word2Vector_vocab.append(vocab_word)
            embed_vector = [float(i) for i in row[1:]]  # convert to list of float
            embedding_dict[vocab_word] = embed_vector
            Word2Vector_embed.append(embed_vector)

        print('Word2Vector Loaded Successfully')
        return Word2Vector_vocab, Word2Vector_embed, embedding_dict
    
Word2Vector_vocab, Word2Vector_embed, embedding_dict = load_word2vector(
    filename='/Users/nzhang/Rebate/embd/sgns.sogounews.bigram-char')


VOCAB_SIZE = 365113
EMBED_SIZE = 300

vocab_list = Word2Vector_vocab[1:]
embed = Word2Vector_embed[1:]


df =pd.DataFrame(embed)
df.fillna(0)

df=df.values


len(vocab_list)
len(embed)
embed[1]

#embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE,EMBED_SIZE],
#                                             minval= -1.0, 
#                                             maxval= 1.0))
embed_matrix  = np.zeros((VOCAB_SIZE, EMBED_SIZE))

len(embed[4])

with open("metadata.tsv", 'w+') as file_metadata:
    for i in range(VOCAB_SIZE):
        embed_matrix[i] = df[i]
        file_metadata.write(vocab_list[i] + '\n')
        

output_path = '/Users/nzhang/Visual/embed/output'

sess = tf.InteractiveSession()

embedding = tf.Variable(embed_matrix, trainable = False, name = 'w2x_metadata')
tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter(output_path, sess.graph)

# adding into projector
config = projector.ProjectorConfig()
embed = config.embeddings.add()
embed.tensor_name = 'metadata'
embed.metadata_path = 'metadata.tsv'

# Specify the width and height of a single thumbnail.
projector.visualize_embeddings(writer, config)
saver.save(sess, os.path.join(output_path,'w2x_metadata.ckpt'))
print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))







embedding_var = tf.Variable(df, name = 'embedding')
sess.run(embedding_var.initializer)
config = projector.ProjectorConfig()
summary_writer = tf.summary.FileWriter(LOGDIR)

# add embeddings to config
embedding = config.embeddings.adds()
embedding.tensor_name = embedding_var.name

 # link the embeddings to their metadata file. In this case, the file that contains
 # the 500 most popular words in our vocabulary
embedding.metadata_path = LOGDIR + '/vocab.tsv'

# save a configuration file that TensorBoard will read during startup
projector.visualize_embeddings(summary_writer, config)

# save our embedding
save_embed = tf.train.Savor([embedding_var])
save_embed.save(sess, LODGDIR + '/skip_gram.ckpt',1)

































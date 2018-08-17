import tensorflow as tf
import numpy as np
import os
import data_helpers
#from tensorflow.contrib import learn
import tflearn
import csv
import codecs
import time
import pandas as pd
import pymysql
from sqlalchemy import create_engine

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "./data/we.txt", "Data source.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string('f', '', 'kernel')


FLAGS = tf.flags.FLAGS

# Load data
sql = 'select url, unicode from flfs_copy limit 2000'
engine = create_engine("mysql+pymysql://root:rzx@1218!@!#@***.***.***.***:51024/funds_task?charset=utf8",echo=True)

data = pd.DataFrame()
for chunks in pd.read_sql_query(sql,con=engine,chunksize=1000):
    data=data.append(chunks)
    

# Map data into vocabulary
vocab_path = os.path.join("./runs/1532276166/", "vocab")
#vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
vocab_processor = tflearn.data_utils.VocabularyProcessor.restore(vocab_path)

text_list=[]
for text in x_raw:
    text_list.append(' '.join(text))
    
x_test = np.array(list(vocab_processor.transform(text_list)))



print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint("./runs/1532276166/checkpoints")
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))

out_path = os.path.join(FLAGS.checkpoint_dir, ".", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with codecs.open(out_path, 'w','utf-8') as f:
    csv.writer(f).writerows(predictions_human_readable)

#Build DataFrame
df = pd.DataFrame({'tag_words':x_raw,'result':all_predictions})
df['update_time']= time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
df['model_version'] = 'V1.0'
df['tag_words']= df['tag_words'].map(lambda s: ' '.join(s))


#type(df['update_time'][0])
type(df['tag_words'][0])

engine = create_engine("mysql+pymysql://root:rzx@1218!@!#@202.108.211.109:51024/funds_task?charset=utf8") 
pd.io.sql.to_sql(df,
                 name='classifer',
                 con=engine,
                 schema= 'funds_task',
                 if_exists='append',
                 chunksize=1000)





    
    

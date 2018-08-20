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
from sqlalchemy import create_engine
import re
import jieba_fast.analyse as jieba

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "", "Data source.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

tf.flags.DEFINE_string('f', '', 'kernel')


FLAGS = tf.flags.FLAGS

stime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(stime)

# Load data
sql = 'select url, unicode from flfs_copy'
engine = create_engine("mysql+pymysql://root:rzx@1218!@!#@202.108.211.109:51024/funds_task?charset=utf8") 

#Process Dataframe 
for data in pd.read_sql_query(sql,con=engine,chunksize=200):
    data['unicode'] = data['unicode'].map(lambda s: re.sub(r'[^\u4E00-\u9FA5]','',str(s)))
    data['tag_words'] = data['unicode'].map(lambda s: jieba.extract_tags(s, withWeight=False, topK=20, allowPOS=('n', 'v', 'nt', 'vn')))
    data.drop('unicode',axis = 1, inplace = True)
    x_raw = data['tag_words'].tolist()

	# Map data into vocabulary
	vocab_path = os.path.join("./runs/1534390388/", "vocab")
	#vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	vocab_processor = tflearn.data_utils.VocabularyProcessor.restore(vocab_path)

	text_list=[]
	for text in x_raw:
	    text_list.append(' '.join(text))
	    
	x_test = np.array(list(vocab_processor.transform(text_list)))


	print("\nEvaluating...\n")

	# Evaluation
	# ==================================================
	checkpoint_file = tf.train.latest_checkpoint("./runs/1534390388/checkpoints")
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


	del text, text_list, vocab_path, x_raw, x_test, x_test_batch, checkpoint_file, batch_predictions

	# Build Dataframe
	data['result'] = pd.Series(all_predictions,name='result')
	data['update_time']= time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	data['model_version'] = 1.5
	data['tag_words']= data['tag_words'].map(lambda s: ' '.join(s))

	#Export to db
	pd.io.sql.to_sql(data,
	                 name='classifer_v',
	                 con=engine,
	                 schema= 'funds_task',
	                 if_exists='append',
	                 chunksize=1000)

endtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
print(endtime)






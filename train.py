import tensorflow as tf
import numpy as np
import tflearn
from sklearn.model_selection import train_test_split
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./data/target.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/non_target.txt", "Data source for the negative data.")
tf.flags.DEFINE_string("pre_trained_Word2vector", "./embd/sgns.sogounews.bigram-char", "Data source for Word2vector.")

# Model Hyper-parameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 300, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    sample_num = len(x_text)

    print('{} samples in x_txt'.format(sample_num))
    print('The min sentence contains ' + str(min([len(x) for x in x_text])))
    print('The max sentence contains ' + str(max([len(x) for x in x_text])))

    # load pre-trained Word2Vector
    Word2Vector_vocab, Word2Vector_embed, embedding_dict = data_helpers.load_word2vector(
        filename=FLAGS.pre_trained_Word2vector)

    # vocab_size = len(Word2Vector_vocab)
    # embedding_dim = len(Word2Vector_embed[0])
    # embedding = np.asarray(Word2Vector_embed)

    print('The dimension of Word2Vector is {}'.format(Word2Vector_embed[0]))
    print('The vocab size is {}'.format(Word2Vector_vocab[0]))

    # Build input array
    # - paddle to same length
    # - create dict and reverse dict with word ids
    max_document_length = max([len(x) for x in x_text])
    vocab_processor = tflearn.data_utils.VocabularyProcessor(max_document_length)
    text_list = []
    for text in x_text:
        text_list.append(' '.join(text))
    x = np.array(list(vocab_processor.fit_transform(text_list)))

    print('The maximum of x is {}'.format(vocab_processor.max_document_length))
    print('The number of vocab in train-set is {}'.format(len(vocab_processor.vocabulary_)))

    """
    Cautious:  embedding is not equal to reverse_dict !!
    
    """

    # Build Embedding array
    doc_vocab_size = len(vocab_processor.vocabulary_)

    # Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping

    # Sort the vocabulary dictionary on the basis of values(id).
    # Both statements perform same task.
    # sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    dict_as_list = sorted(vocab_dict.items(), key=lambda x: x[1])

    embeddings_tmp = []

    for i in range(doc_vocab_size):
        item = dict_as_list[i][0]
        if item in Word2Vector_vocab:
            embeddings_tmp.append(embedding_dict[item])
        else:
            rand_num = np.random.uniform(low=-0.2, high=0.2, size=FLAGS.embedding_dim)
            embeddings_tmp.append(rand_num)

    # final embedding array corresponds to dictionary of words in the document
    embedding = np.asarray(embeddings_tmp)

    print('The shape of embedding is '.format(embedding.shape))

    # Randomly shuffle data
    x_shuffled, y_shuffled = tflearn.data_utils.shuffle(x, y)

    print('The size of y_shuffled is {}'.format(len(y_shuffled)))
    print('The size of x_shuffled is {}'.format(len(x_shuffled)))

    # Split train-set and test-set
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size = 0.2, random_state=0)


    print('The size of train-set is {}'.format(len(x_train)))
    print('The size of test-set is {}'.format(len(x_dev)))

    del x, y, x_shuffled, y_shuffled, embeddings_tmp

    return x_train, y_train, vocab_processor, x_dev, y_dev, embedding


def train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            init = tf.global_variables_initializer()

            sess.run(cnn.embedding_init, feed_dict={cnn.embedding_placeholder: embedding})
            sess.run(init)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, embedding= preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, embedding)


if __name__ == '__main__':
    tf.app.run()



import numpy as np
import tensorflow as tf
import random
from dataloader import ABC_Data_Loader
# from data_loader import Data_loader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
import pickle
import abc_reader
from tensorflow.python.platform import gfile
from time import strftime, localtime


FLAGS = tf.app.flags.FLAGS
######################################################################################
#  ABC Reader related
######################################################################################
tf.app.flags.DEFINE_string('TOKEN_MODE', 'GUITAR_CHORD', 'There are three modes for creating dictionary: SINGLE_CHAR  / DISTINCT_SCALE / GUITAR_CHORD.'
                            'SINGLE_CHAR encodes representations as sequence of single characters (so low pitches like C_ are encoded separately. DISTINCE_SCALE considers pitches and flat/sharps'
                           'GUITAR_CHORD creates expanded vocab set from DISTINCT_SCALE mode. It adds guitar chords like "C" to vocabulary set')
tf.app.flags.DEFINE_boolean('HEADER_AS_VOCA', True, 'If set True, headers (ex: M:4/4, K:D) are added to vocabulary set')

######################################################################################
#  General variables
######################################################################################
SEED = 88
melody_size = 83 # will be decided later

tf.app.flags.DEFINE_integer('SEQ_LENGTH',120, 'Sequence Length')
tf.app.flags.DEFINE_integer('START_TOKEN',0, 'Start token for generating samples from generator')
tf.app.flags.DEFINE_string('log_dir', 'log/seqgan_experimient-log1.txt', 'logpath')
tf.app.flags.DEFINE_integer('sample_num',40, 'Number of samples when generating')
tf.app.flags.DEFINE_string('pretrain_ckpt_dir','save/train/pretrain_single_larger/', 'pre-train checkpoint directory')
tf.app.flags.DEFINE_string('rollout_ckpt_dir','save/train/rollout/', 'rollout checkpoint directory')


######################################################################################
#  Generator  Hyper-parameters
######################################################################################
tf.app.flags.DEFINE_integer('GEN_EMB_DIM',32, 'dimension of embedding layer')
tf.app.flags.DEFINE_integer('GEN_HIDDEN_DIM',30,'dimension of hidden layer')
tf.app.flags.DEFINE_integer('GEN_PRE_EPOCH_NUM',0, 'Number of epoches for pretraining')
tf.app.flags.DEFINE_integer('GEN_BATCH_SIZE',32, 'Batch size of generator')

#########################################################################################
#  Discriminator  Hyper-parameters (currently not used)
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  RL Training parameters
#########################################################################################
tf.app.flags.DEFINE_string('RL_ITER_NUM',10, 'Iteration number of RL')
tf.app.flags.DEFINE_float('RL_update_rate',0.8, 'Update rate of RL')

# (not used currerntly)
dis_num_epochs = 1
dis_alter_epoch = 25



def generate_samples(sess, trainable_model, batch_size, generated_num, output_dir, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    # file_name = 'target_generate/pretrain_small.pkl'
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    with open(output_dir + output_file, 'w') as fout:
        pickle.dump(generated_samples, fout)

def target_loss(sess, target_lstm, data_loader):
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():

    ABC_READER = abc_reader.ABC_Reader(FLAGS.TOKEN_MODE, FLAGS.HEADER_AS_VOCA, FLAGS.SEQ_LENGTH, 'abc/mnt.txt', 'abc/mnt_converted.txt')
    melody_size, note_dict_path, tr_data_path = ABC_READER.create_dict()


    random.seed(SEED)
    np.random.seed(SEED)

    gen_data_loader = ABC_Data_Loader(FLAGS.GEN_BATCH_SIZE)
    likelihood_data_loader = ABC_Data_Loader(FLAGS.GEN_BATCH_SIZE) # For testing

    # dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(melody_size, FLAGS.GEN_BATCH_SIZE, FLAGS.GEN_EMB_DIM, FLAGS.GEN_HIDDEN_DIM, FLAGS.SEQ_LENGTH, FLAGS.START_TOKEN)
    # discriminator = Discriminator(sequence_length=64, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
    #                             filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4))
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(tr_data_path)

    print strftime("%Y-%m-%d %H:%M:%S", localtime())
    log = open('save/experiment-log.txt', 'w')

    #  # pre-train generator
    # print 'Start pre-training...'
    # log.write('pre-training...\n')
    # for epoch in xrange(FLAGS.GEN_PRE_EPOCH_NUM):
    #     loss = pre_train_epoch(sess, generator, gen_data_loader)
    #
    #     if epoch % 10 == 0:
    #         print 'pre-train epoch ', epoch, 'test_loss ', loss
    #         buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(loss) + '\n'
    #         log.write(buffer)
    #         if epoch % 50 == 0:
    #             file_dir = 'target_generate/pretrain_single_larger/'
    #             file_name = 'pretrain_epoch' + str(epoch) + '.pkl'
    #             generate_samples(sess, generator, FLAGS.GEN_BATCH_SIZE, FLAGS.sample_num, file_dir, file_name)
    #             likelihood_data_loader.create_batches(file_dir+file_name)
    #             if not gfile.Exists(FLAGS.pretrain_ckpt_dir):
    #                 gfile.MakeDirs(FLAGS.pretrain_ckpt_dir)
    #             generator.save_variables(sess, FLAGS.pretrain_ckpt_dir, epoch)

    generator.restore_variables(sess, FLAGS.pretrain_ckpt_dir)

    # print 'Start pre-training discriminator...'
    # # Train 3 epoch on the generated data and do this for 50 times
    # for _ in range(50):
    #     generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
    #     dis_data_loader.load_train_data(positive_file, negative_file)
    #     for _ in range(3):
    #         dis_data_loader.reset_pointer()
    #         for it in xrange(dis_data_loader.num_batch):
    #             x_batch, y_batch = dis_data_loader.next_batch()
    #             feed = {
    #                 discriminator.input_x: x_batch,
    #                 discriminator.input_y: y_batch,
    #                 discriminator.dropout_keep_prob: dis_dropout_keep_prob
    #             }
    #             _ = sess.run(discriminator.train_op, feed)


    rollout = ROLLOUT(generator, ABC_READER, FLAGS.RL_update_rate)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    print strftime("%Y-%m-%d %H:%M:%S", localtime())
    for total_batch in range(FLAGS.RL_ITER_NUM):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards, tot_err, tot_wrn = rollout.get_reward(sess, samples, 8, FLAGS.SEQ_LENGTH)
            buffer = 'Iter:\t' + str(total_batch) + '\tTotal Error:\t' + str(tot_err) + '\tTotal Warning:\t' + str(tot_wrn) + '\n'
            print buffer
            log.write(buffer)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)



        # Test
        if total_batch % 5 == 0 or total_batch == FLAGS.RL_ITER_NUM - 1:
            file_dir = 'target_generate/'
            file_name = 'rollout_epoch' + str(total_batch) + '.pkl'
            generate_samples(sess, generator, FLAGS.GEN_BATCH_SIZE, FLAGS.sample_num, file_dir, file_name)
            likelihood_data_loader.create_batches(file_dir+file_name)
            test_loss = target_loss(sess, generator, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            log.write(buffer)
            if not gfile.Exists(FLAGS.rollout_ckpt_dir):
                gfile.MakeDirs(FLAGS.rollout_ckpt_dir)
            generator.save_variables(sess, FLAGS.rollout_ckpt_dir, total_batch)

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        # for _ in range(5):
        #     generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
        #     dis_data_loader.load_train_data(positive_file, negative_file)
        #
        #     for _ in range(3):
        #         dis_data_loader.reset_pointer()
        #         for it in xrange(dis_data_loader.num_batch):
        #             x_batch, y_batch = dis_data_loader.next_batch()
        #             feed = {
        #                 discriminator.input_x: x_batch,
        #                 discriminator.input_y: y_batch,
        #                 discriminator.dropout_keep_prob: dis_dropout_keep_prob
        #             }
        #             _ = sess.run(discriminator.train_op, feed)

    log.close()
    print strftime("%Y-%m-%d %H:%M:%S", localtime())


if __name__ == '__main__':
    main()

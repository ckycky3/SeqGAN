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

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 320 # hidden state dimension of lstm cell # 32 -> 320
SEQ_LENGTH = 64 # sequence length
START_TOKEN = 48
PRE_EPOCH_NUM = 200 # supervise (maximum likelihood estimation) epochs
# PRE_EPOCH_NUM = 0 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 32

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 10

dis_num_epochs = 1
dis_alter_epoch = 25

positive_file = 'save/abc_trans.pkl'
negative_file = 'target_generate/pretrain_small.pkl'
# eval_file = 'target_generate/midi_trans_eval.pkl'
logpath = 'log/seqgan_experimient-log1.txt'
generated_num = 40

ckpt_path = 'save/train/pretrain_single_larger/'

melody_size = 83
RL_update_rate = 0.8


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    # file_name = 'target_generate/pretrain_small.pkl'
    with open(output_file, 'w') as fout:
        pickle.dump(generated_samples, fout)

def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
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
    ABC_READER = abc_reader.ABC_Reader()
    ABC_READER.create_dict()


    random.seed(SEED)
    np.random.seed(SEED)

    gen_data_loader = ABC_Data_Loader(BATCH_SIZE)
    likelihood_data_loader = ABC_Data_Loader(BATCH_SIZE) # For testing

    # dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(melody_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    # discriminator = Discriminator(sequence_length=64, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim,
    #                             filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    # generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)

    log = open('save/experiment-log.txt', 'w')
     # pre-train generator
    print 'Start pre-training...'
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        loss = pre_train_epoch(sess, generator, gen_data_loader)

        if epoch % 50 == 0:
            file_name = 'target_generate/pretrain_single_larger/pretrain_epoch' + str(epoch) + '.pkl'
            generate_samples(sess, generator, BATCH_SIZE, generated_num, file_name)
            likelihood_data_loader.create_batches(file_name)

            print 'pre-train epoch ', epoch, 'test_loss ', loss
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(loss) + '\n'
            log.write(buffer)

    generator.save_variables(sess, ckpt_path)

    # generator.restore_variables(sess, ckpt_path)

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


    # rollout = ROLLOUT(generator, ABC_READER, 0.8)
    #
    # print '#########################################################################'
    # print 'Start Adversarial Training...'
    # log.write('adversarial training...\n')
    # for total_batch in range(TOTAL_BATCH):
    #     # Train the generator for one step
    #     for it in range(1):
    #         samples = generator.generate(sess)
    #         rewards = rollout.get_reward(sess, samples, 2)
    #         feed = {generator.x: samples, generator.rewards: rewards}
    #         _ = sess.run(generator.g_updates, feed_dict=feed)
    #
    #     # Test
    #     if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
    #         file_name = 'target_generate/pretrain_epoch' + str(total_batch) + '.pkl'
    #         generate_samples(sess, generator, BATCH_SIZE, generated_num, file_name)
    #         likelihood_data_loader.create_batches(file_name)
    #         test_loss = target_loss(sess, generator, likelihood_data_loader)
    #         buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
    #         print 'total_batch: ', total_batch, 'test_loss: ', test_loss
    #         log.write(buffer)
    #
    #     # Update roll-out parameters
    #     rollout.update_params()

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


if __name__ == '__main__':
    main()

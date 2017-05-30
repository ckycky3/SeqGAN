import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import numpy as np
import abc2midi.abc_errorcheck as abc_errorcheck
from abc_reader import ABC_Reader
import os
import re
import subprocess


class ROLLOUT(object):
    def __init__(self, lstm, ABC_Reader, update_rate):
        self.lstm = lstm
        self.ABC_Reader = ABC_Reader
        self.update_rate = update_rate

        self.num_emb = self.lstm.num_emb
        self.batch_size = self.lstm.batch_size
        self.emb_dim = self.lstm.emb_dim
        self.hidden_dim = self.lstm.hidden_dim
        self.sequence_length = self.lstm.sequence_length
        self.start_token = tf.identity(self.lstm.start_token)
        self.learning_rate = self.lstm.learning_rate

        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.create_recurrent_unit()  # maps h_tm1 to h_t for generator
        self.g_output_unit = self.create_output_unit()  # maps h_t to o_t (output token logits)

        self.abc2midi_path = os.path.join('abc2midi', 'bin', 'abc2midi')
        #####################################################################################################
        # placeholder definition
        self.x = tf.placeholder(tf.int32, shape=[self.batch_size, self.sequence_length]) # sequence of tokens generated by generator
        self.given_num = tf.placeholder(tf.int32)

        # processed for batch
        with tf.device("/cpu:0"):
            self.processed_x = tf.transpose(tf.nn.embedding_lookup(self.g_embeddings, self.x), perm=[1, 0, 2])  # seq_length x batch_size x emb_dim

        ta_emb_x = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self.sequence_length)
        ta_emb_x = ta_emb_x.unstack(self.processed_x)

        ta_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length)
        ta_x = ta_x.unstack(tf.transpose(self.x, perm=[1, 0]))
        #####################################################################################################

        self.h0 = tf.zeros([self.batch_size, self.hidden_dim])
        self.h0 = tf.stack([self.h0, self.h0])

        gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.sequence_length,
                                             dynamic_size=False, infer_shape=True)

        # When current index i < given_num, use the provided tokens as the input at each time step
        def _g_recurrence_1(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            x_tp1 = ta_emb_x.read(i)
            gen_x = gen_x.write(i, ta_x.read(i))
            return i + 1, x_tp1, h_t, given_num, gen_x

        # When current index i >= given_num, start roll-out, use the output as time step t as the input at time step t+1
        def _g_recurrence_2(i, x_t, h_tm1, given_num, gen_x):
            h_t = self.g_recurrent_unit(x_t, h_tm1)  # hidden_memory_tuple
            o_t = self.g_output_unit(h_t)  # batch x vocab , logits not prob
            log_prob = tf.log(tf.nn.softmax(o_t))
            next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
            x_tp1 = tf.nn.embedding_lookup(self.g_embeddings, next_token)  # batch x emb_dim
            gen_x = gen_x.write(i, next_token)  # indices, batch_size
            return i + 1, x_tp1, h_t, given_num, gen_x

        i, x_t, h_tm1, given_num, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, given_num, _4: i < given_num,
            body=_g_recurrence_1,
            loop_vars=(tf.constant(0, dtype=tf.int32),
                       tf.nn.embedding_lookup(self.g_embeddings, self.start_token), self.h0, self.given_num, gen_x))

        _, _, _, _, self.gen_x = control_flow_ops.while_loop(
            cond=lambda i, _1, _2, _3, _4: i < self.sequence_length,
            body=_g_recurrence_2,
            loop_vars=(i, x_t, h_tm1, given_num, self.gen_x))

        self.gen_x = self.gen_x.stack()  # seq_length x batch_size
        self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

    def get_reward(self, sess, input_x, rollout_num, seq_length):
        rewards = []
        header = '''X:1
'''
        tot_err = 0
        tot_wrn = 0
        for i in range(rollout_num):
            print "Rollout #", i
            for given_num in range(1, seq_length):
                print given_num
                feed = {self.x: input_x, self.given_num: given_num}
                samples = sess.run(self.gen_x, feed)

                # TODO: Modify get_reward
                # feed = {discriminator.input_x: samples, discriminator.dropout_keep_prob: 1.0}
                # ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
                # ypred = np.array([item[1] for item in ypred_for_auc])
                decode_samples = self.ABC_Reader.trans_trans_songs_to_raw(samples)
                # print "samples to one abc"
                one_abc = self.samples_to_one_abc(decode_samples, header)
                # print one_abc
                # print "abc to message"
                integrated_message = self.abc_to_message(one_abc)
                # print integrated_message
                # print "message to error list"
                error_list = self.message_to_error_list(integrated_message, False, tot_err, tot_wrn)
                reward = np.array([0 if error else 1 for error in error_list])
                # for sample in decode_samples:
                #     # print "-------------Sample---------------"
                #     # print sample
                #     has_error = self.error_check(sample, abc2midi_path, header, False)
                #     error_list.append(has_error)
                #     reward = np.array([0 if error else 1 for error in error_list])
                if i == 0:
                    rewards.append(reward)
                else:
                    rewards[given_num - 1] += reward

            # the last token reward
            # feed = {discriminator.input_x: input_x, discriminator.dropout_keep_prob: 1.0}
            # ypred_for_auc = sess.run(discriminator.ypred_for_auc, feed)
            # ypred = np.array([item[1] for item in ypred_for_auc])
            decode_samples = self.ABC_Reader.trans_trans_songs_to_raw(input_x)
            one_abc = self.samples_to_one_abc(decode_samples, header)
            integrated_message = self.abc_to_message(one_abc)
            error_list, tot_err, tot_wrn = self.message_to_error_list(integrated_message, True, tot_err, tot_wrn)
            reward = np.array([0 if error else 1 for error in error_list])
            # error_list = []
            # for sample in decode_samples:
            #     has_error = self.error_check(sample, abc2midi_path, header, True)
            #     error_list.append(has_error)
            #     reward = np.array([0 if error else 1 for error in error_list])
            if i == 0:
                rewards.append(reward)
            else:
                rewards[seq_length-1] += reward

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
        return rewards, tot_err, tot_wrn

    def create_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = tf.identity(self.lstm.Wi)
        self.Ui = tf.identity(self.lstm.Ui)
        self.bi = tf.identity(self.lstm.bi)

        self.Wf = tf.identity(self.lstm.Wf)
        self.Uf = tf.identity(self.lstm.Uf)
        self.bf = tf.identity(self.lstm.bf)

        self.Wog = tf.identity(self.lstm.Wog)
        self.Uog = tf.identity(self.lstm.Uog)
        self.bog = tf.identity(self.lstm.bog)

        self.Wc = tf.identity(self.lstm.Wc)
        self.Uc = tf.identity(self.lstm.Uc)
        self.bc = tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def update_recurrent_unit(self):
        # Weights and Bias for input and hidden tensor
        self.Wi = self.update_rate * self.Wi + (1 - self.update_rate) * tf.identity(self.lstm.Wi)
        self.Ui = self.update_rate * self.Ui + (1 - self.update_rate) * tf.identity(self.lstm.Ui)
        self.bi = self.update_rate * self.bi + (1 - self.update_rate) * tf.identity(self.lstm.bi)

        self.Wf = self.update_rate * self.Wf + (1 - self.update_rate) * tf.identity(self.lstm.Wf)
        self.Uf = self.update_rate * self.Uf + (1 - self.update_rate) * tf.identity(self.lstm.Uf)
        self.bf = self.update_rate * self.bf + (1 - self.update_rate) * tf.identity(self.lstm.bf)

        self.Wog = self.update_rate * self.Wog + (1 - self.update_rate) * tf.identity(self.lstm.Wog)
        self.Uog = self.update_rate * self.Uog + (1 - self.update_rate) * tf.identity(self.lstm.Uog)
        self.bog = self.update_rate * self.bog + (1 - self.update_rate) * tf.identity(self.lstm.bog)

        self.Wc = self.update_rate * self.Wc + (1 - self.update_rate) * tf.identity(self.lstm.Wc)
        self.Uc = self.update_rate * self.Uc + (1 - self.update_rate) * tf.identity(self.lstm.Uc)
        self.bc = self.update_rate * self.bc + (1 - self.update_rate) * tf.identity(self.lstm.bc)

        def unit(x, hidden_memory_tm1):
            previous_hidden_state, c_prev = tf.unstack(hidden_memory_tm1)

            # Input Gate
            i = tf.sigmoid(
                tf.matmul(x, self.Wi) +
                tf.matmul(previous_hidden_state, self.Ui) + self.bi
            )

            # Forget Gate
            f = tf.sigmoid(
                tf.matmul(x, self.Wf) +
                tf.matmul(previous_hidden_state, self.Uf) + self.bf
            )

            # Output Gate
            o = tf.sigmoid(
                tf.matmul(x, self.Wog) +
                tf.matmul(previous_hidden_state, self.Uog) + self.bog
            )

            # New Memory Cell
            c_ = tf.nn.tanh(
                tf.matmul(x, self.Wc) +
                tf.matmul(previous_hidden_state, self.Uc) + self.bc
            )

            # Final Memory cell
            c = f * c_prev + i * c_

            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(c)

            return tf.stack([current_hidden_state, c])

        return unit

    def create_output_unit(self):
        self.Wo = tf.identity(self.lstm.Wo)
        self.bo = tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_output_unit(self):
        self.Wo = self.update_rate * self.Wo + (1 - self.update_rate) * tf.identity(self.lstm.Wo)
        self.bo = self.update_rate * self.bo + (1 - self.update_rate) * tf.identity(self.lstm.bo)

        def unit(hidden_memory_tuple):
            hidden_state, c_prev = tf.unstack(hidden_memory_tuple)
            # hidden_state : batch x hidden_dim
            logits = tf.matmul(hidden_state, self.Wo) + self.bo
            # output = tf.nn.softmax(logits)
            return logits

        return unit

    def update_params(self):
        self.g_embeddings = tf.identity(self.lstm.g_embeddings)
        self.g_recurrent_unit = self.update_recurrent_unit()
        self.g_output_unit = self.update_output_unit()

    def sample_to_abc(self, sample):
        string = ''
        for char in sample:
            string += char
        return string

    def samples_to_one_abc(self, samples, header):
        large_abc = ''
        enter = '''

'''
        separator = '''=======================

'''
        for sample in samples:
            large_abc += header
            abc = self.sample_to_abc(sample)
            large_abc += abc
            large_abc += enter
            large_abc += header
            large_abc += separator

        return large_abc

    def abc_to_message(self, abc):
        abc = abc.replace(r'\"', ' ') # replace escaped " characters with space since abc2midi doesn't understand them
        lines = re.split('\r\n|\r|\n', abc)
        for i in range(len(lines)):
            if lines[i].startswith('X:'):
                line = lines[i]
                del lines[i]
                lines.insert(0, line)
                break
        abc_code = os.linesep.join([l.strip() for l in lines])

        old_stress_model = any(l for l in lines if l.startswith('%%MIDI stressmodel 1'))
        creationflags = 0
        cmd = [self.abc2midi_path, '-', '-c']
        if not old_stress_model:
            cmd = cmd + ['-BF']
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=creationflags)
        stdout_value, stderr_value = process.communicate(input=(abc_code + os.linesep * 2).encode('latin-1', 'ignore'))
        if stdout_value:
            stdout_value = re.sub(r'(?m)(writing MIDI file .*\r?\n?)', '', stdout_value)
        return stdout_value

    def message_to_error_list(self, message, show, err, wrn):
        error_list = []
        separator = False
        lines = re.split('\r\n|\r|\n', message)
        err_cnt = 0
        wrn_cnt = 0
        if show:
            print message
        for i in range(len(lines)):
            if "Ignoring text: =======================" in lines[i]:
                separator = True
                # print "Ignoring text found", i
            elif separator and "No valid K: field found at start of tune" in lines[i]:
                if show:
                    print err_cnt, " Errors, ", wrn_cnt, " Warnings"
                if err_cnt == 0 and wrn_cnt == 0:
                    error_list.append(False)
                else:
                    error_list.append(True)
                separator = False
                err_cnt = 0
                wrn_cnt = 0
            elif lines[i].startswith('Error'):
                if "No R: in header, cannot apply Barfly model" in lines[i]:
                    continue
                else:
                    if show:
                        print lines[i]
                        err += 1
                    err_cnt += 1
            elif lines[i].startswith('Warning'):
                if show:
                    print lines[i]
                    wrn += 1
                wrn_cnt += 1
        # print len(error_list)
        if show:
            return error_list, err, wrn
        else:
            return error_list

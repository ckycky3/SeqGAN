import numpy as np
import pandas as pd
import pickle

class ABC_Reader:

    def __init__(self):
        self.note_info_path = 'abc_mapping_dict.pkl'
        self.midi_training_path_trans = "save/abc_trans.pkl"

    def preprocess(self):

        bad_words=['T:','%','X:','S:','A:','B:','C:','D:','F:','G:','H:','I:','O:','r:','U:','W:','w:','Z:']

        with open('abc/mnt.txt') as oldfile, open('abc/mnt_converted.txt', 'w') as newfile:
            for line in oldfile:
                if not any(bad_word in line for bad_word in bad_words):
                    if not line.strip() == '':
                        newfile.write(line)


    def create_dict(self):

        scales_extended = ['C,','D,','E,','G,','A,','B,', "c'","d'","e'","f'","g'","a'","b'"]
        scales = ['C','D','E','F','G','A','B','c','d','e','f','g','a','b']

        fh = open('abc/mnt_converted.txt').read()

        print fh
        unique_chars = set(fh)
        length = 64
        sorted_vals = map(str, unique_chars)
        sorted_vals = np.asarray(sorted_vals)
        note_info = pd.DataFrame(data = sorted_vals, columns=['note'])
        print len(unique_chars)
        print note_info[note_info['note'] == 'M']

        self.note_info_dict = note_info['note'].to_dict()
        self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.iteritems())

        with open(self.note_info_path, "w") as openfile:
            pickle.dump(self.note_info_dict, openfile)

        print len(fh)

        trans_fh = []
        for cha in fh:
            new_cha = self.note_info_dict_swap.get(cha)
            trans_fh.append(new_cha)



        trans_list = []

        last_index = 0

        while last_index + length < len(trans_fh):
            trans_list.append(trans_fh[last_index:last_index + length])
            last_index += length

        with open(self.midi_training_path_trans, "w") as output_file:
            pickle.dump(trans_list, output_file)

    def trans_generated_to_midi(self, file_name):

        with open(file_name + ".pkl", 'rb') as files:
            res = pickle.load(files)
            print res

        raws = self.trans_trans_songs_to_raw(res)

        index = 0
        for raw in raws:
            print"========================="
            print ''.join(raw)

    def trans_trans_songs_to_raw(self, trans_list):
        raw_list = []
        for midi in trans_list:
            raw_list.append(np.asarray(self.trans_to_raw_note(midi)))

        return raw_list

    def trans_to_raw_note(self, trans_note):

        result = []

        for entry in trans_note:
            result.append(self.note_info_dict.get(entry))

        return result


if __name__ == "__main__":
    reader = ABC_Reader()
    reader.create_dict()
    # reader.trans_generated_to_midi('pretrain_epoch100')

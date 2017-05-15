import numpy as np
import pandas as pd
import pickle
import re

class ABC_Reader:

    def __init__(self):
        self.note_info_path = 'abc_mapping_dict.pkl'
        self.midi_training_path_trans = "save/abc_trans.pkl"
        # SINGLE_CHAR  / DISTINCT_SCALE / GUITAR_CODE
        self.mode = 'DISTINCT_SCALE'

    def preprocess(self):

        bad_words=['T:','%','X:','S:','A:','B:','C:','D:','F:','G:','H:','I:','O:','r:','U:','W:','w:','Z:']

        with open('abc/mnt.txt') as oldfile, open('abc/mnt_converted.txt', 'w') as newfile:
            for line in oldfile:
                if not any(bad_word in line for bad_word in bad_words):
                    if not line.strip() == '':
                        newfile.write(line)


    def create_dict(self):
        capital_scales = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        lower_scales = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
        normal_scales = capital_scales + lower_scales

        scales_extended = []
        capital_postfix = ','
        lower_postfix = '\''

        for scale in capital_scales:
            scales_extended.append(scale+capital_postfix)
        for scale in lower_scales:
            scales_extended.append(scale+lower_postfix)

        sharp_flat_prefix = ['^','_','=']
        sharp_flatten_scales = []
        for prefix in sharp_flat_prefix:
            for scales in normal_scales+scales_extended:
                newchar = prefix+scales
                # print newchar
                sharp_flatten_scales.append(newchar)

        fh = open('abc/mnt_converted.txt').read()

        # background_filtered = re.findall(r'\"^[\w\d].{1,2}\"',fh)
        # background_filtered = re.findall(r'\"[A-Z0-9].{1,3}\"', fh)
        # print background_filtered

        unique_chars = set(fh)
        length = 64
        sorted_vals = map(str, unique_chars)
        sorted_vals+=scales_extended
        sorted_vals+=sharp_flatten_scales
        sorted_vals = np.asarray(sorted_vals)


        note_info = pd.DataFrame(data = sorted_vals, columns=['note'])

        # print note_info[note_info['note'] == 'M']

        self.note_info_dict = note_info['note'].to_dict()
        self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.iteritems())

        with open(self.note_info_path, "w") as openfile:
            pickle.dump(self.note_info_dict, openfile)



        if self.mode == 'SINGLE_CHAR':
            trans_fh = []
            char_buffer = []
            for char in fh:
                reschars = []

                if len(char_buffer) != 0:
                    # if string is not in dict, convert character individually
                    # if is in dict, check for postfix
                    char_buffer += char

                    if not self.list_to_char(char_buffer) in self.note_info_dict_swap.itervalues():
                        reschars = char_buffer
                        char_buffer = []
                    else:
                        if len(char_buffer) == 3:
                            if not self.list_to_char(char_buffer) in self.note_info_dict_swap.itervalues():
                                prev_chars = self.list_to_chat(char_buffer[0:2])
                                new_char = self.list_to_char(char_buffer[2])

                                reschars += prev_chars
                                reschars += new_char
                                char_buffer = []

                            else:
                                reschars = self.list_to_char(char_buffer)
                                char_buffer = []

                else:
                    if not self.is_char_in_list(char, normal_scales + sharp_flat_prefix):
                        reschars += char
                    else:
                        char_buffer += char

                if not len(reschars) == 0:
                    for char in reschars:
                        new_cha = self.note_info_dict_swap.get(char)
                        trans_fh.append(new_cha)

        elif self.mode == 'DISTINCT_SCALE':
            trans_fh = []
            for char in fh:
                trans_fh.append(self.note_info_dict_swap.get(char))


        trans_list = []

        last_index = 0

        while last_index + length < len(trans_fh):
            trans_list.append(trans_fh[last_index:last_index + length])
            last_index += length

        with open(self.midi_training_path_trans, "w") as output_file:
            pickle.dump(trans_list, output_file)




    def list_to_char(self, list):
        return ''.join(map(str, list))
    def is_char_in_list(self, char, list):
        if any(cha in char for cha in list):
            return True

        return False

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

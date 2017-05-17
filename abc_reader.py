import numpy as np
import pandas as pd
import pickle
import re
from collections import OrderedDict

class ABC_Reader:

    def __init__(self):
        self.note_info_path = 'abc_mapping_dict.pkl'
        self.midi_training_path_trans = "save/abc_trans.pkl"
        # SINGLE_CHAR  / DISTINCT_SCALE / GUITAR_CHORD
        self.mode = 'GUITAR_CHORD'
        self.window_length = 64

    def preprocess(self):
        # bad_words=['T:','%','X:','S:','A:','B:','C:','D:','F:','G:','H:','I:','O:','r:','U:','W:','w:','Z:']
        bad_words = ['T:', '%', 'X:', 'S:', 'A:', 'B:', 'C:', 'D:', 'F:', 'G:', 'H:', 'I:', 'O:', 'r:', 'U:', 'W:',
                     'w:', 'Z:', 'M:','L:','K:','P:','N:','R:']

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


        guitar_chord_prefix = '\"'
        guitar_chords = []

        trans_fh = []
        reschars = []
        if self.mode == 'SINGLE_CHAR':
            reschars = fh
        else:
            filter_out_list = []
            scale_prefix = normal_scales+sharp_flat_prefix

            if self.mode == 'DISTINCT_SCALE':
                filter_out_list = scale_prefix
            elif self.mode == 'GUITAR_CHORD':
                filter_out_list = scale_prefix+list(guitar_chord_prefix)


            trans_fh = []
            char_buffer = []
            buffer_mode = ''
            for char in fh:

                if len(char_buffer) != 0:
                    # if string is not in dict, convert character individually
                    # if is in dict, check for postfix
                    char_buffer += char


                    if buffer_mode == 'scale':

                        if not self.list_to_char(char_buffer) in scales_extended+sharp_flatten_scales:

                            if not (self.mode == 'GUITAR_CHORD' and char == guitar_chord_prefix):
                                reschars += char_buffer
                                buffer_mode = ''
                                char_buffer = []
                            else:
                                reschars += char_buffer[:-1]
                                buffer_mode = 'guitar'
                                char_buffer = char_buffer[-1:]



                        else:
                            if len(char_buffer) == 3:
                                if not self.list_to_char(char_buffer) in scales_extended+sharp_flatten_scales:
                                    prev_chars = self.list_to_char(char_buffer[0:2])
                                    new_char = self.list_to_char(char_buffer[2])

                                    reschars += prev_chars
                                    reschars += new_char
                                    buffer_mode = ''

                                    char_buffer = []

                                else:
                                    reschars += self.list_to_char(char_buffer)
                                    buffer_mode = ''

                                    char_buffer = []

                    elif buffer_mode == 'guitar':

                        if char == guitar_chord_prefix:
                            chord = self.list_to_char(char_buffer)
                            reschars += chord

                            if not (chord in guitar_chords):
                                guitar_chords.append(chord)

                            buffer_mode = ''
                            if len(chord) > 4:
                                print chord

                            char_buffer = []

                else:
                    if not self.is_char_in_list(char, filter_out_list):
                        reschars += char
                    else:
                        char_buffer += char

                        if self.is_char_in_list(char, scale_prefix):
                            buffer_mode = 'scale'

                        if self.mode == 'GUITAR_CHORD':
                            if char == guitar_chord_prefix:
                                buffer_mode = 'guitar'



        unique_chars = set(fh)
        sorted_vals = map(str, unique_chars)

        if self.mode == 'DISTINCT_SCALE' or self.mode == 'GUITAR_CHORD':
            sorted_vals+=scales_extended
            sorted_vals+=sharp_flatten_scales





        if self.mode == 'GUITAR_CHORD':
            sorted_vals+=guitar_chords

        sorted_vals = np.asarray(sorted_vals)
        note_info = pd.DataFrame(data = sorted_vals, columns=['note'])
        self.note_info_dict = note_info['note'].to_dict()
        self.note_info_dict_swap = dict((y, x) for x, y in self.note_info_dict.iteritems())

        with open(self.note_info_path, "w") as openfile:
            pickle.dump(self.note_info_dict, openfile)


        for char in reschars:
            new_cha = self.note_info_dict_swap.get(char)
            trans_fh.append(new_cha)

        print trans_fh[:50]

        trans_list = []
        last_index = 0
        while last_index + self.window_length < len(trans_fh):
            trans_list.append(trans_fh[last_index:last_index + self.window_length])
            last_index += self.window_length

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
    reader.preprocess()
    reader.create_dict()
    # reader.trans_generated_to_midi('pretrain_epoch100')

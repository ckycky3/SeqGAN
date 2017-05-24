import os, os.path
import sys
import re
import subprocess
import hashlib
from fraction import Fraction

try:
    import pygame
    import pygame.pypm as pypm
    pypm.Initialize()
except ImportError:
    try:
        import pypm
    except ImportError:
        sys.stderr.write('Warning: pygame/pypm module not found. Recording midi will not work')


def abc_check(abc_code, header, abc2midi_path, play_chords=False, default_midi_program=1, tempo_multiplier=None, add_meta_data=False):
    # add some extra lines to header (whether chords are on/off and what default midi program to use for each channel) 
    extra_lines = []
    for channel in range(1, 16+1):
        extra_lines.append('%%%%MIDI program %d %d' % (channel, default_midi_program))
    if play_chords:
        extra_lines.append('%%MIDI gchordon')
    else:
        extra_lines.append('%%MIDI gchordoff')
    header = os.linesep.join(extra_lines + [header.strip()])
    
    abc_code = process_abc_code(abc_code, header, tempo_multiplier=tempo_multiplier, minimal_processing=True)
    abc_code = abc_code.replace(r'\"', ' ')  # replace escaped " characters with space since abc2midi doesn't understand them    

    # print abc_code

    # make sure that X field is on the first line since abc2midi doesn't seem to support
    # fields and instructions that come before the X field
    lines = re.split('\r\n|\r|\n', abc_code)
    for i in range(len(lines)):
        if lines[i].startswith('X:'):
            line = lines[i]
            del lines[i]
            lines.insert(0, line)
            break
    abc_code = os.linesep.join([l.strip() for l in lines])
    print abc_code

    old_stress_model = any(l for l in lines if l.startswith('%%MIDI stressmodel 1'))

    creationflags = 0

    # abc2midi_path = 'abc2midi'
    # midi_file = os.path.abspath(os.path.join(cache_dir, 'temp_%s.midi' % hash))
    # print "midi_file : ", midi_file
    # if not os.path.exists(midi_file):

    cmd = [abc2midi_path, '-', '-c']
    if not old_stress_model:
        cmd = cmd + ['-BF']
    #if add_meta_data:
    #    cmd = cmd + ['-EA']
    # print cmd
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags = creationflags)
    stdout_value, stderr_value = process.communicate(input=(abc_code+os.linesep*2).encode('latin-1', 'ignore'))
    # if "Error" in stdout_value:
    #     print "Error"
    # if "Warning" in stdout_value:
    #     print "Warning"
    # print "stderr_value : ", stderr_value
    proc = subprocess.Popen(['sleep', str(0.5)])
    proc.communicate()

    if stdout_value:
        stdout_value = re.sub(r'(?m)(writing MIDI file .*\r?\n?)', '', stdout_value)
    if process.returncode != 0:
        return None

    #if humanize:
    #    humanize_midi(midi_file, midi_file)
    #    pass
    # process.terminate()
    return stdout_value


def process_abc_code(abc_code, header, minimal_processing=False, tempo_multiplier=None, landscape=False):
    ''' adds file header and possibly some extra fields, and may also change the Q: field '''

    extra_lines = \
        '%%leftmargin 0.5cm\n' \
        '%%rightmargin 0.5cm\n' \
        '%%botmargin 0cm\n' \
        '%%topmargin 0cm\n'

    if minimal_processing:
        extra_lines = ''
    parts = []
    if landscape and not minimal_processing:
        parts.append('%%landscape 1\n')
    if header:
        parts.append(header.rstrip() + os.linesep)
    if extra_lines:
        parts.append(extra_lines)
    parts.append(abc_code)
    abc_code = ''.join(parts)

    abc_code = re.sub(r'\[\[(.*/)(.+?)\]\]', r'\2', abc_code)  # strip PmWiki links and just include the link text
    if tempo_multiplier:
        abc_code = change_abc_tempo(abc_code, tempo_multiplier)

    abc_code = process_MCM(abc_code)

    return abc_code


def change_abc_tempo(abc_code, tempo_multiplier):
    ''' multiples all Q: fields in the abc code by the given multiplier and returns the modified abc code '''

    def subfunc(m, multiplier):
        try:
            if '=' in m.group(0):
                parts = m.group(0).split('=')
                parts[1] = str(int(int(parts[1]) * multiplier))
                return '='.join(parts)

            q = int(int(m.group(1)) * multiplier)
            if '[' in m.group(0):
                return '[Q: %d]' % q
            else:
                return 'Q: %d' % q
        except:
            return m.group(0)

    abc_code, n1 = re.subn(r'(?m)^Q: *(.+)', lambda m, mul=tempo_multiplier: subfunc(m, mul), abc_code)
    abc_code, n2 = re.subn(r'\[Q: *(.+)\]', lambda m, mul=tempo_multiplier: subfunc(m, mul), abc_code)
    # if no Q: field that is not inline add a new Q: field after the X: line
    # (it seems to be ignored by abcmidi if added earlier in the code)
    if n1 == 0:
        default_tempo = 120
        extra_line = 'Q:%d' % int(default_tempo * tempo_multiplier)
        lines = re.split('\r\n|\r|\n', abc_code)
        for i in range(len(lines)):
            if lines[i].startswith('X:'):
                lines.insert(i + 1, extra_line)
                break
        abc_code = os.linesep.join(lines)
    return abc_code


def process_MCM(abc):
    abc, n = re.subn(r'(?m)^(L:\s*mcm_default)', r'L:1/8', abc)
    if n:
        # erase non-note fragments of the text by replacing them by spaces (thereby preserving offsets)
        repl_by_spaces = lambda m: ' ' * len(m.group(0))
        s = abc.replace('\r', '\n')
        s = re.sub(r'(?s)\%\%begin(ps|text).+?\%\%end(ps|text)', repl_by_spaces,
                   s)  # remove embedded text/postscript
        s = re.sub(r'(?m)^\w:.*?$|%.*$', repl_by_spaces, s)  # remove non-embedded fields and comments
        s = re.sub(r'".*?"|!.+?!|\+\w+?\+|\[\w:.*?\]', repl_by_spaces,
                   s)  # remove strings, ornaments and embedded fields

        fragments = []
        last_fragment_end = 0
        for m in re.finditer(r"(?P<note>([_=^]?[A-Ga-gxz](,+|'+)?))(?P<len>\d{0,2})(?P<dot>\.?)", s):
            if m.group('len') == '':
                length = 0
            else:
                length = Fraction(8, int(m.group('len')))
                if m.group('dot'):
                    length = length * 3 / 2

            start, end = m.start(0), m.end(0)
            fragments.append((False, abc[last_fragment_end:start]))
            fragments.append((True, m.group('note') + str(length)))
            last_fragment_end = end
        fragments.append((False, abc[last_fragment_end:]))
        abc = ''.join((text for is_note, text in fragments))
    return abc


if __name__ == '__main__':

    abc = '''X:1
M:3/4
L:1/16
K:Ddor
C2G,C C2E2 C2E2 | C4 C2G,C C2B,2 | C4 G,6A,A, | C2B,2 C2E2 C4 | C4 C2B,2 C4 |
G,4>G,4 A,2B,2 | B,2D2 G,8 | E2F2 E4 E6 |C2 E2E2 G4 |
G4 C4 D2D2 | D4 D4 |'''
#     abc = '''X:1
# M:3/4
# L:1/16
# K:Ddor
# laksejoifheqiuugh3wiueqriojwef'''
    abc2midi_path = os.path.join('abc2midi', 'bin', 'abc2midi')
    midi = abc_check(abc_code=abc, header='', abc2midi_path=abc2midi_path)
    print midi
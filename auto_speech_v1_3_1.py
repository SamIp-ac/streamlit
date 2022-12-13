import music21
from music21 import *
import pandas as pd

# Commented out IPython magic to ensure Python compatibility.
import gtts
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd
import librosa, librosa.display
import time
"""#### Read the .mxl file"""


def auto_speech(filename):
    with open('temp_mxl.mxl', "wb") as f:
        f.write(filename.getbuffer())

    c = music21.converter.parse('temp_mxl.mxl')  # Input the song/piece here.

    note_list = []
    duration_list = []

    for n in c.flat.notesAndRests:
        try:

            temp_name = n.pitch.name
            if n.duration.quarterLength == float(0.0):
                temp_name = str('Ornaments ') + str(n.pitch.name)

            note_list.append(str(temp_name) + str(n.pitch.octave))
            duration_list.append(n.duration.quarterLength)
        except:
            try:
                for i in n:
                    None

                temp_note = []
                temp_duration = []

                for i in n:
                    temp_note.append(str(i.pitch.name) + str(i.pitch.octave))
                    temp_duration.append(i.duration.quarterLength)

                y = ''
                for i in temp_note:
                    y += str(i)
                    y = y + ','

                # Added
                x = ''
                for i in temp_duration:
                    x += str(i)
                    x = x + ','

                note_list.append(str('chord, ') + y + str('chord end'))
                duration_list.append(str(x[:-1]))

            except:

                try:
                    note_list.append('Rest')
                    duration_list.append(n.duration.quarterLength)
                except:
                    raise 'It is not a note, chord or rest'

    """#### Make it into data frame"""

    pd.set_option('display.max_rows', 1500)
    df = pd.DataFrame()
    df['Note type'] = note_list
    df.insert(1, 'duration (in s)', duration_list)

    """#### Convert duration to the name of notes' type"""

    duration_dictionary = {float(0.0625): 'note 64th', float(0.125): 'note 32nd', float(0.25): 'note 16th',
                           float(0.5): 'note 8th', float(1.0): 'Quarter', float(2.0): 'Half', float(4.0): 'Whole',
                           float(0.0): 'Ornament', float(1.5): 'Quarter with augmentation Dot',
                           float(2.5): 'Half with augmentation Dot',
                           float(0.75): 'note 16th with augmentation Dot'}  # Need to add more cases
    duration_list_converted = []

    for i in range(len(duration_list)):

        try:
            k = []
            temp = duration_dictionary[duration_list[i]]
            duration_list_converted.append(temp)

        except:
            k = []
            temp_2 = []
            k.append(duration_list[i].split(','))

            for j in k:
                for h in j:
                    h = float(h)
                    temp_2.append(duration_dictionary[float(h)])

            x = ''
            for j in temp_2:
                x = x + str(j) + ','
            duration_list_converted.append(x[:-1])

    df.insert(2, 'Note duration name', duration_list_converted)

    """#### Finding the key, time signature and metrome mark from .mxl file"""

    key = c.analyze('key')
    meta_key = str(key)

    dict_meta = dict()
    dict_meta['key'] = meta_key

    TimeSignature = c.recurse().getTimeSignatures()[0]
    meta_TimeSignature = str(TimeSignature)
    dict_meta['TimeSignature'] = meta_TimeSignature.split(' ')[-1][:-1].split('/')[0] + ' ' + \
                                 meta_TimeSignature.split(' ')[-1][:-1].split('/')[-1]

    metronomeMark = c.metronomeMarkBoundaries()[0][2]
    metronomeMark = str(metronomeMark)
    dict_meta['metronomeMark'] = metronomeMark.split(' ')[-1][:-1]

    tempo = dict_meta['metronomeMark']
    Time_Signature = dict_meta['TimeSignature']
    keys = dict_meta['key']

    """#### Convert it into audio and save as .wav file/ .mp3 file

  """

    a = note_list
    b = duration_list_converted

    plt.rcParams['figure.figsize'] = (13, 5)

    # conbine two lists in alternation
    result = [None] * (len(a) + len(b))
    result[::2] = a
    result[1::2] = b

    output = ''

    output = 'tempo, ' + tempo + ', time signature, ' + Time_Signature + ', keys, ' + keys + ', '

    # list = 423*2 = [:846]
    for i in result[816:]:
        output += str(i)
        output = output + '... '

    t1 = gtts.gTTS(output, slow=False)
    # save the audio file
    t1.save('pages/Data/temp_autospeech.wav')

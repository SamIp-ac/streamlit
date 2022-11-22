import music21
import pandas as pd


def mxl2uni(filename):

    c = music21.converter.parse(filename)  # Input the song/piece here.

    '''for i in c.getTimeSignatures():
        print(i)
    print(c)'''
    # show(c)

    '''print('------')'''
    for i in c:
        print(i)

    '''print(c.metadata.all())'''

    # Not used
    tempo = 120
    time_sig = .5  # ratio to 4-times note
    ###################
    note_list = []
    duration_list = []

    # print(c.recurse().notes)

    # If it is a chord
    '''for i in c.flat.notes[347]:
      print("Note: %s%d %0.01f" % (i.pitch.name, i.pitch.octave, i.duration.quarterLength))'''
    for n in c.flat.notesAndRests:
        try:
            # print("Note: %s%d %0.01f" % (n.pitch.name, n.pitch.octave, n.duration.quarterLength))
            temp_name = n.pitch.name
            if n.duration.quarterLength == float(0.0):  # it is a Ornaments
                temp_name = str('Ornaments ') + str(n.pitch.name)  # O for Ornaments

            note_list.append(str(temp_name) + str(n.pitch.octave))
            duration_list.append(n.duration.quarterLength)  # *(60/tempo)/time_sig
        except:
            try:
                for i in n:
                    None
                # print('------------------It is a chord:------------------')
                temp_note = []
                temp_duration = []

                for i in n:
                    # print("Note: %s%d %0.01f" % (i.pitch.name, i.pitch.octave, i.duration.quarterLength))
                    temp_note.append(str(i.pitch.name) + str(i.pitch.octave))
                    temp_duration.append(i.duration.quarterLength)  # *(60/tempo)/time_sig
                # print('------------------------End------------------------')

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
                duration_list.append(str(x[:-1]))  # Changed: del last ','

            except:
                # print("Rest Note: %0.01f" % (n.duration.quarterLength))
                try:
                    note_list.append('Rest')
                    duration_list.append(n.duration.quarterLength)  # *(60/tempo)/time_sig
                except:
                    raise 'It is not a note, chord or rest'

    # show(c)
    all_list = zip(note_list, duration_list)

    pd.set_option('display.max_rows', 1500)
    df = pd.DataFrame()
    df['Note type'] = note_list
    df.insert(1, 'duration (in s)', duration_list)

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

    '''print(duration_list_converted)'''

    df.insert(2, 'Note duration name', duration_list_converted)

    key = c.analyze('key')
    meta_key = str(key)

    dict_meta = dict()
    dict_meta['key'] = meta_key

    TimeSignature = c.recurse().getTimeSignatures()[0]
    meta_TimeSignature = str(TimeSignature)
    dict_meta['TimeSignature'] = meta_TimeSignature.split(' ')[-1][:-1][0] + meta_TimeSignature.split(' ')[-1][:-1][-1]

    metronomeMark = c.metronomeMarkBoundaries()[0][2]
    metronomeMark = str(metronomeMark)
    dict_meta['metronomeMark'] = metronomeMark.split(' ')[-1][:-1]

    '''print(dict_meta)'''

    tempo = dict_meta['metronomeMark']
    Time_Signature = dict_meta['TimeSignature']
    keys = dict_meta['key']

    """Grouping into bar"""

    ...

    """Unicode"""

    # Start
    temp_symbol = ['5-line staff',
                   'Single barline', 'Final barline',
                   'G clef',
                   'metronome whole', 'metronome half', 'metronome quarter', 'metronome note 8th',
                   'metronome note 16th']
    temp_symbol_unicode = ['E01A',
                           'E030', 'E032',
                           'E050',
                           'ECA2', 'ECA3', 'ECA5', 'ECA7', 'ECA9']

    temp_symbol_dict = dict(zip(temp_symbol, temp_symbol_unicode))

    # Note name
    unicode_6 = ['EB90', 'EB91', 'EB92', 'EB93', 'EB94', 'EB95', 'EB96', 'EB97', 'EB91',  # D6 'E511\u' + 'EB91'
                 '', 'EB98', 'EB99', 'EB9A', 'EB9B', 'EB9C', 'EB9D', 'EB9E', 'EB9F']  # B4
    unicode_6_name = ['C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5', 'C6', 'D6',  # D6
                      'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4', 'B3', 'A3']  # B4

    Note_name_dict = dict(zip(unicode_6_name, unicode_6))

    # Note type
    unicode_7 = ['E1D2', 'E1D3',
                 'E1D5',
                 'E1D7',
                 'E1D9']
    unicode_7_name = ['Whole', 'Half',
                      'Quarter',
                      'note 8th',
                      'note 16th']

    unicode_77 = ['E1D2', 'E1D4', 'E1D6', 'E1D8', 'E1DA']
    unicode_77_name = ['Whole', 'Half', 'Quarter', 'note 8th', 'note 16th']

    Note_type_down_dict = dict(zip(unicode_7_name, unicode_7))
    Note_type_up_dict = dict(zip(unicode_77_name, unicode_77))
    # beam
    unicode_8 = ['E8E0', 'E8E1',  # Only for 'note 8th', 'note 16th'
                 'E8E2', 'E8E3']
    unicode_8_name = ['Begin beam', 'End beam',
                      'Begin tie', 'End tie']

    beam_dict = dict(zip(unicode_8_name, unicode_8))
    # Rest
    # if == rest
    unicode_9 = ['E4E0', 'E4E1', 'E4E2', 'E4E3', 'E4E4', 'E4E5', 'E4E6', 'E4E7', 'E4E8']
    unicode_9_name = ['restMaxima', 'restLonga', 'restDoubleWhole', 'Whole', 'Half', 'Quarter', 'note 8th', 'note 16th',
                      'note 32nd']
    ############# First 3 have not change name

    rest_dict = dict(zip(unicode_9_name, unicode_9))
    # Time signature
    unicode_10 = ['E081', 'E082', 'E083', 'E084', 'E085', 'E086', 'E087', 'E088', 'E089']
    temp_Time_signature = ['E09E', 'E09F']
    unicode_10_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    Time_signature_dict = dict(zip(unicode_10_name, unicode_10))

    unicode_11_name = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    unicode_11 = ['', 2, '', 1, '', '', '', .5, '']
    Time_signature_duration_dict = dict(zip(unicode_11_name, unicode_11))

    unicode_12_name = ['A5', 'B5', 'C6', 'D6', 'E6',  # E5 position of legar has problem
                       'C4', 'B3', 'A3', 'G3']
    unicode_12 = ['EB95\\u' + 'E022', 'EB95\\u' + 'E022', 'EB95\\u' + 'E022\\u' + 'EB97\\u' + 'E022',
                  'EB95\\u' + 'E022\\u' + 'EB97\\u' + 'E022', 'EB95\\u' + 'E022\\u' + 'EB97\\u' + 'E022',
                  'EB9D\\u' + 'E022', 'EB9D\\u' + 'E022', 'EB9D\\u' + 'E022\\u' + 'EB9F\\u' + 'E022',
                  'EB9D\\u' + 'E022\\u' + 'EB9F\\u' + 'E022']
    Legar_dict = dict(zip(unicode_12_name, unicode_12))

    '''print(rest_dict)'''

    Unicode = []
    clef = 'G clef'
    '''print(df.head())'''
    temp_symbol_dict['5-line staff']

    beam_checking_list = ['note 8th', 'note 16th']
    note_check_list = ['A5', 'B5', 'C6', 'D6', 'E6',
                       'C4', 'B3', 'A3', 'G3']

    # 5-line staff
    Unicode.append(temp_symbol_dict['5-line staff'])
    # clef
    Unicode.append(temp_symbol_dict[clef])

    # Time signature
    Unicode.append(
        temp_Time_signature[0] + '\\u' + Time_signature_dict[dict_meta['TimeSignature'][0]] + '\\u' +
        temp_Time_signature[
            -1] + '\\u' + Time_signature_dict[dict_meta['TimeSignature'][-1]])
    Unicode.append(temp_symbol_dict['5-line staff'])
    Unicode.append('2004')

    base_unit = Time_signature_duration_dict[str(dict_meta['TimeSignature'][-1])]
    counting = 0

    for i in range(df.shape[0]):

        counting += float(df.iloc[i, 1])
        Unicode.append(temp_symbol_dict['5-line staff'])  # add 5-line

        if len(str(df.iloc[i, 0])) == 3:
            df.iloc[i, 0] = df.iloc[i, 0][0] + df.iloc[i, 0][-1]

        if df.iloc[i, 0] == 'Rest':
            Unicode.append(rest_dict[df.iloc[i, 2]])
            # blank
            Unicode.append('2004')

        elif str(df.iloc[i, 0])[:9] == 'Ornaments':
            None
        elif len(str(df.iloc[i, 0])) == 2:

            for j in range(len(unicode_6_name)):
                if unicode_6_name[j] == str(df.iloc[i, 0]):
                    temp__ = j

            if int(temp__) < int(9):

                '''if df.iloc[i,2] in beam_checking_list and df.iloc[i + 1,2] in beam_checking_list: # add beam
                  Unicode.append(beam_dict['Begin beam'])'''

                Unicode.append(Note_name_dict[df.iloc[i, 0]] + '\\u' + Note_type_up_dict[df.iloc[i, 2]])

                if df.iloc[i, 0] in note_check_list:  # need external lines
                    Unicode.append(Legar_dict[df.iloc[i, 0]])

            elif int(temp__) > int(9):

                Unicode.append(Note_name_dict[df.iloc[i, 0]] + '\\u' + Note_type_down_dict[df.iloc[i, 2]])

            elif int(temp__) == int(9):  # special case for B4
                if df.iloc[i, 2] not in beam_checking_list:
                    Unicode.append(Note_type_up_dict[df.iloc[i, 2]])  # B4 up vs down ???????????

                elif df.iloc[i, 2] in beam_checking_list:
                    Unicode.append(Note_type_up_dict[df.iloc[i, 2]])
            # blank
            Unicode.append('2004')

        while counting == int(dict_meta['TimeSignature'][0]) * base_unit:
            try:
                temp___ = df.iloc[i + 1, 0]
                Unicode.append(temp_symbol_dict['Single barline'])
                Unicode.append(temp_symbol_dict['5-line staff'])
                Unicode.append('2004')
            except:
                Unicode.append(temp_symbol_dict['5-line staff'])
                Unicode.append('2004')

            counting = 0

    Unicode.append(temp_symbol_dict['Final barline'])

    all_ = '\\u'

    for i in range(len(Unicode)):
        all_ = all_ + Unicode[i] + '\\u'

    all_ = all_[:-3]

    txt = all_

    x = txt.split('\\')

    st = ''

    for i in x:
        st += i + '\\'

    return st

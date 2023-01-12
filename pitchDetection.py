import librosa
import pandas as pd
import pickle
import numpy as np


def message(someText):
    print('-------------%s-------------' % someText)


tempList = [['E2', '82.41'],  # A=440Hz
            ['F2', '87.31'],
            ['F#2/Gb2', '92.5'],
            ['G2', '98'],
            ['G#2/Ab2', '103.83'],
            ['A2', '110'],
            ['A#2/Bb2', '116.54'],
            ['B2', '123.47'],
            ['C3', '130.81'],
            ['C#3/Db3', '138.59'],
            ['D3', '146.83'],
            ['D#3/Eb3', '155.56'],
            ['E3', '164.81'],
            ['F3', '174.61'],
            ['F#3/Gb3', '185'],
            ['G3', '196'],
            ['G#3/Ab3', '207.65'],
            ['A3', '220'],
            ['A#3/Bb3', '233.08'],
            ['B3', '246.94'],
            ['C4', '261.63'],
            ['C#4/Db4', '277.18'],
            ['D4', '293.66'],
            ['D#4/Eb4', '311.13'],
            ['E4', '329.63'],
            ['F4', '349.23'],
            ['F#4/Gb4', '369.99'],
            ['G4', '392'],
            ['G#4/Ab4', '415.3'],
            ['A4', '440'],
            ['A#4/Bb4', '466.16'],
            ['B4', '493.88'],
            ['C5', '523.25'],
            ['C#5/Db5', '554.37'],
            ['D5', '587.33'],
            ['D#5/Eb5', '622.25'],
            ['E5', '659.25'],
            ['F5', '698.46'],
            ['F#5/Gb5', '739.99'],
            ['G5', '783.99'],
            ['G#5/Ab5', '830.61'],
            ['A5', '880.00'],
            ['A#5/Bb5', '932.33'],
            ['B5', '987.77'],
            ['C6', '1046.50'],
            ['C#6/Db6', '1108.73'],
            ['D6', '1174.66'],
            ['D#6/Eb6', '1244.51'],
            ['E6', '1318.51'],
            ['F6', '1396.91'],
            ['F#6/Gb6', '1479.98'],
            ['G6', '1567.98']]

freq2pitch = dict()
for i in tempList:
    tempDict = {float(i[1]): i[0]}
    freq2pitch.update(tempDict)

freq2pitch.update({float(0): 'Rest'})


def voting(List):  # Not use now
    return max(set(List), key=List.count)


def whole_pitch(whole_array, sampling_rate_=44100):
    message('Loading')
    whole_array = np.array(whole_array, dtype=float)

    if sampling_rate_ == 44100:
        frame_length_ = 4404  # 2047 * 2 4404
    else:
        frame_length_ = 2202  # 2048

    ms2s_unit = 0.1  # duration (in s) for each frame_length 0.093
    temp = len(whole_array) // frame_length_

    remainder = 0
    if len(whole_array) / frame_length_ != temp:
        remainder = len(whole_array) - (frame_length_ * temp)

    piece_array = []

    message('Cutting')
    for i in range(0, temp):
        piece_array.append(whole_array[i * frame_length_:(i + 1) * frame_length_])

    piece_array.append(whole_array[-remainder:])

    noteList = []
    errorList = []
    durationList = []

    foundation_freqList = []

    message('Analysing and converting')
    for k in piece_array:
        f0_, voiced_flag, voiced_probs = librosa.pyin(k,
                                                      frame_length=frame_length_,
                                                      fill_na=0,
                                                      fmin=librosa.note_to_hz('C2'),
                                                      fmax=librosa.note_to_hz('C7'),
                                                      sr=sampling_rate_)
        # print(f0_)
        if len(f0_) >= 4:
            f0_ = f0_[1:5]

        error_ = 100
        thePitch_ = 'Rest'

        for j in freq2pitch.keys():
            if abs(j - (sum(f0_) / (len(f0_)))) <= abs(error_):
                error_ = (j - (sum(f0_) / (len(f0_))))
                thePitch_ = freq2pitch[j]

        noteList.append(thePitch_)
        errorList.append(error_)
        durationList.append(float(ms2s_unit))

        foundation_freqList.append((sum(f0_) / (len(f0_))))

    durationList.pop()
    durationList.append(ms2s_unit * remainder / frame_length_)

    assert len(noteList) == len(errorList) == len(durationList)

    tempNoteList = [str(noteList[0])]
    tempDurationList = [durationList[0]]
    tempErrorList = [errorList[0]]

    message('Compressing')
    count = 1
    for h in range(len(noteList) - 1):

        if str(noteList[h + 1]) == str(tempNoteList[-1]):
            count += 1
            tempDurationList[-1] += durationList[h + 1]
            tempErrorList[-1] += errorList[h + 1]

        elif str(noteList[h + 1]) != str(tempNoteList[-1]):
            tempErrorList[-1] /= count
            tempNoteList.append(str(noteList[h + 1]))
            tempDurationList.append(durationList[h + 1])
            tempErrorList.append(errorList[h + 1])
            count = 1

    message('Listing')
    ResultList = list(zip(tempNoteList, tempDurationList, tempErrorList))

    df = pd.DataFrame()

    df['noteName'] = tempNoteList
    df['duration (in s)'] = tempDurationList
    df['error'] = tempErrorList

    message('Done')
    return tempNoteList, tempDurationList, tempErrorList, foundation_freqList, df


'''file = open('items.txt', 'w')
for item in y[:40000]:
    file.write(str(item) + ",")
file.close()'''
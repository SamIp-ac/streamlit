import music21
from music21 import *
from midi2audio import FluidSynth
import os


class mxl2wav:

    def __init__(self):
        self.stage = 'end'

    def mxl2midi(self, filename, output_filename, instruments, switch=True):

        with open('temp_mxl.mxl', "wb") as f:
            f.write(filename.getbuffer())
        """Input: file need to convert, output name (no need .mid)"""
        c = music21.converter.parse('temp_mxl.mxl')
        if switch:
            for el in c.recurse():
                if 'Instrument' in el.classes:
                    if instruments == 'violin':
                        el.activeSite.replace(el, instrument.Violin())
                    elif instruments == 'piano':
                        el.activeSite.replace(el, instrument.Piano())
                    elif instruments == 'flute':
                        el.activeSite.replace(el, instrument.Flute())
                    elif instruments == 'clarinet':
                        el.activeSite.replace(el, instrument.Clarinet())

        fp = c.write('midi', output_filename + '.mid')

    def midi2wav(self, input_filename, output_filename):
        """Input: file need to convert, output name (no need .wav)"""
        fs = FluidSynth('font.sf2', sample_rate=44100)  # must need .sf2 file
        fs.midi_to_audio(input_filename, output_filename + '.wav')


def m2w(mxlfile, instruments):
    mw = mxl2wav()
    mw.mxl2midi(mxlfile, 'temp_midi', instruments)
    mw.midi2wav('temp_midi' + '.mid', 'pages/Data/temp_mxl2wav')



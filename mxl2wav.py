import music21
from music21 import *
from midi2audio import FluidSynth


class mxl2wav:

    def __init__(self):
        self.stage = 'end'

    def mxl2midi(self, filename, output_filename, instrument, switch=True):
        """Input: file need to convert, output name (no need .mid)"""
        c = music21.converter.parse(filename)
        if switch:
            for el in c.recurse():
                if 'Instrument' in el.classes:
                    if instrument == 'violin':
                        el.activeSite.replace(el, instrument.Violin())
                    elif instrument == 'piano':
                        el.activeSite.replace(el, instrument.Piano())

        fp = c.write('midi', output_filename + '.mid')

    def midi2wav(self, input_filename, output_filename):
        """Input: file need to convert, output name (no need .wav)"""
        fs = FluidSynth('font.sf2', sample_rate=44100)  # must need .sf2 file
        fs.midi_to_audio(input_filename, output_filename + '.wav')


def m2w(mxlfile, instrument):
    mw = mxl2wav()
    mw.mxl2midi(mxlfile, 'temp_midi', instrument)
    mw.midi2wav('temp_midi' + 'mid', 'temp')


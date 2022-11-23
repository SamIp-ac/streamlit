import music21
from music21 import *
from midi2audio import FluidSynth
from IPython.display import Image, Audio
import mido
import IPython
import pandas as pd
from mido import MidiFile, MidiFile, MidiTrack
import librosa
import librosa.display
import numpy as np
import IPython.display as ipd


class mxl2wav:

    def __init__(self):
        self.stage = 'end'

    def mxl2midi(self, filename, output_filename, switch=False):
        """Input: file need to convert, output name (no need .mid)"""
        c = music21.converter.parse(filename)
        if switch:
            for el in c.recurse():
                if 'Instrument' in el.classes:
                    el.activeSite.replace(el, instrument.Violin())

        fp = c.write('midi', output_filename + '.mid')

    def midi2wav(self, input_filename, output_filename):
        """Input: file need to convert, output name (no need .wav)"""
        fs = FluidSynth('font.sf2', sample_rate=44100)  # must need .sf2 file
        fs.midi_to_audio(input_filename, output_filename + '.wav')


def m2w(mxlfile):
    mw = mxl2wav()


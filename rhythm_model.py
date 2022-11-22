from scipy.io import wavfile
import numpy as np
from scipy import signal
import librosa
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import datasets, mixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import librosa.display
from sklearn.cluster import DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import davies_bouldin_score
from scipy.io.wavfile import write
import os


class Audio:  # REMARK: for violin only
    def __init__(self):
        self.stage = 2

    def read_audio(self, file_name, type_='wavfile', sampling_rate=44100):
        '''Read file convert to data'''
        if type_ == 'librosa':
            x, sr = librosa.load(file_name, sr=sampling_rate)
            x = librosa.to_mono(x)

        elif type_ == 'wavfile':
            sr, x = wavfile.read(file_name)

        sr = sampling_rate
        return x, sr

    def preprocessing_signal(self, list_of_filename, range_vio=20):
        '''
        return: np.array, np.array.shape, frequencies, times
        last two use for plot spectrogram
        extract files convert to spectrogram and concatenate it to a list for machine learning
        time is different, so fill zero to it. (padding)
        '''
        X_train = []
        temp = []
        times_list = []
        fs_list = []
        new_X_train = []
        for files in list_of_filename:
            fs, data = wavfile.read(files)  # fs mostly is 44100Hz, but it is 22050Hz in librosa
            if data.shape[1] == 2:
                x = (data[:, 0] + data[:, 1]) / 2  # for handling stereo recording
                # If all device for recording is stereo recording, a new model structure can be induced.
            else:
                x = data[:, 0]

            frequencies, times, spectrogram = signal.spectrogram(x, fs)
            # freq and time maybe different for different recording device
            # Trun it to a violin's freq range
            self.frequencies, spectrogram = frequencies[1:1 + range_vio], spectrogram[1:1 + range_vio,
                                                                          :]  # Really useful??
            X_train.append(spectrogram)
            times_list.append(times)
            fs_list.append(fs)
        for i in X_train:
            temp.append(np.array(i).shape[1])

        max = np.max(temp)
        position = np.where(temp == np.max(temp))[0]

        # Zero padding
        for i in X_train:
            temps = np.zeros((range_vio, max))
            temps[0:np.array(i).shape[0], 0:np.array(i).shape[1]] = np.array(i)
            new_X_train.append(temps)

        # Reset the times to match the spectrogram, as the spectrogram cahnged
        for i in position:
            self.times_t = times_list[i]
            self.fs_t = fs_list[i]

        # times = temp[position]
        return np.array(new_X_train), np.array(new_X_train).shape, self.frequencies, self.times_t, self.fs_t

    def spectrogram_librosa(self, list_of_filename, sampling_rate=44100,
                            win_length=2048, n_fft=2048):
        '''Generate mel spectrogram by using librosa
        Output: audio data, spectrogram, sampling rate'''

        X_train_audio = []
        X_train_spectrogram = []

        if type(list_of_filename[0]) == str:
            for files in list_of_filename:
                x, sr = librosa.load(files, sr=sampling_rate)  # 22050 defalt
                X_train_audio.append(x)
                S = librosa.feature.melspectrogram(x, sr=sr)
                # Change unit to dB
                # S_dB = librosa.power_to_db(S, ref=np.max)
                X_train_spectrogram.append(S)

        else:

            S = librosa.feature.melspectrogram(list_of_filename, sr=sampling_rate
                                               , win_length=win_length,
                                               n_fft=2048)  # smaller = time clear, larger = freq clear
            # Change unit to dB
            # S_dB = librosa.power_to_db(S, ref=np.max)
            X_train_audio.append(list_of_filename)
            X_train_spectrogram.append(S)
            sr = sampling_rate

        return (X_train_audio), (X_train_spectrogram), sr

    def rhythm_extract(self, a_spectrogram, critical_value=2.5, mode=1, pitch=12):
        # Long note problem
        '''
        mode 1 is return all main data to 1
        elif 'original': return original amplitude
        elif 'specific': number of specific frequency only to be kept
        Extraction of the rhythm.
        Problem: too many for-loop, only use on one spectrogram, so you may need for-loop outside
        '''
        pitch2freq_dict = {}  # need to edit
        list_of_max = []
        b = np.array(a_spectrogram)

        # Denoise part (deep learning later?)
        for j in range(0, b.shape[1]):
            c = b[:, j]
            d = np.where(c == np.max(c))
            list_of_max.append(d)

        list_of_max = np.array(list_of_max)

        # rhythm extraction
        # Convert to 1
        if mode == 1:
            # changed
            m = np.mean(b)
            sd = np.std(b)

            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):

                    if b[i, j] >= m + critical_value * sd:
                        b[i, j] = 1
                    else:
                        b[i, j] = 0

            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):
                    if i != list_of_max[j]:
                        b[i, j] = 0
                # Shift ?
            m = np.mean(b)
            sd = np.std(b)

            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):

                    if b[i, j] >= m + critical_value * sd:
                        b[i, j] = 1
                    else:
                        b[i, j] = 0
        # Unchange
        elif mode == 'original':
            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):
                    if i != list_of_max[j]:
                        b[i, j] = 0
                # Shift ?
            m = np.mean(b)
            sd = np.std(b)

            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):

                    if b[i, j] < m + critical_value * sd:
                        b[i, j] = 0

        elif mode == 'specific':
            temp = np.sum(b, axis=1)
            temp_list = []

            m = np.mean(b)
            sd = np.std(b)

            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):

                    if b[i, j] < m + critical_value * sd:
                        b[i, j] = 0

            for i in range(pitch):
                temp_1 = np.argmax(temp)
                temp_list.append(temp_1)
                temp[temp_1] = 0

            m = np.mean(b)
            sd = np.std(b)

            for i in range(0, b.shape[0]):
                for j in range(0, b.shape[1]):

                    if b[i, j] < m + critical_value * sd:
                        b[i, j] = 0

            c = np.zeros((b.shape[0], b.shape[1]))
            c[temp_list, :] = b[temp_list, :]

            for i in range(0, c.shape[0]):
                for j in range(0, c.shape[1]):

                    if c[i, j] > 0:
                        c[i, j] = 1
            b = c

        a_spectrogram = b

        return a_spectrogram

    def rhythm_extract_2(self, a_spectrogram, critical_value=7):
        # Long note problem
        '''
        Extraction of the rhythm.
        Problem: too many for-loop, only use on one spectrogram, so you may need for-loop outside
        '''

        b = np.array(a_spectrogram)
        m = np.mean(b)
        sd = np.std(b)
        list_of_max = []

        # Denoise part (use deep learning later?)
        for j in range(0, b.shape[1]):
            c = b[:, j]
            d = np.where(c == np.max(c))
            list_of_max.append(d)

        list_of_max = np.array(list_of_max)

        # rhythm extraction
        for i in range(0, b.shape[0]):
            for j in range(0, b.shape[1]):
                # Shift ?

                if i != list_of_max[j]:
                    b[i, j] = 0
                if b[i, j] >= m + critical_value * sd:
                    b[i, j] = 1
                else:
                    b[i, j] = 0

        a_spectrogram = b

        return a_spectrogram

    def rhythm_extract_specific(self, a_spectrogram, list_of_note, critical_value=7):
        '''Only counting the specific frequency'''

        # Context

        b = np.array(a_spectrogram)
        m = np.mean(b)
        sd = np.std(b)
        list_of_max = []

    def converter(self, a_spectrogram):
        '''Convert it into 1D list, maybe useful and easy to view.
        Suggest to use after the hythm extract procedure'''

        e5 = np.sum(a_spectrogram, axis=0)

        for i in range(0, len(e5)):
            if e5[i] >= 1:
                e5[i] = 1

        return e5

    def bar_cutting(self, recording, total_bar, tempo_of_setting, Time_Signatures,
                    sampling_rate=44100):
        # The cutting is not so accurate.
        '''Cut a recording into bar units, the input should include a whole ready beats

        Input: a audio data (NOT file name), no. of total bar, tempo set by user,
        Time_Signatures (fill 4 if Time Signatures is 4/4), sampling rate

        Output: a list of audio data (No a file name now)
        '''

        self.total_bar = total_bar
        list_of_data = []

        # Padding the recording to prevent out of range
        recordings = np.array(recording).reshape(-1, 1)
        # Normalized
        min_max_scaler = MinMaxScaler()
        recordings = min_max_scaler.fit_transform(recordings)
        # Find initial point
        initial = np.where(abs(recordings) >= (min_max_scaler.transform(np.array([0]).reshape(-1, 1)) * 1.01)[0][0])[0][
            0]
        recordings = np.squeeze(recordings[initial:])  # (n,1) to (n,)

        temps = np.zeros((recordings.shape[0] + int(
            np.round_(4 * Time_Signatures * sampling_rate * 60 / tempo_of_setting)),))  # extensed 4 bar
        temps[:recordings.shape[0]] = recordings
        recording = temps

        a = range(0, total_bar + 1)  # With ready beats (one bar)
        Hz = sampling_rate  # 22050 Hz or 44100 Hz

        seconds_per_beat = 60 / tempo_of_setting
        #
        no_of_data = int(np.round_(Time_Signatures * Hz * seconds_per_beat))  # no. of data for a bar

        for i in a:
            temp = recording[
                   no_of_data * i: no_of_data * i + no_of_data + no_of_data * 3]  # To be decided # int(np.round_(seconds_per_beat))
            list_of_data.append(temp)

        # list_of_data = list_of_data[1:]

        return list_of_data, sampling_rate

    def tempo_extractor(self, a_bar, sampling_rate=44100):
        '''Finding the tempo of a bar, for-loop may be needed outside after bar_cutting function
        Input: a_bar should be either audio data or a file of a bar
        Output: global tempo, sound wave with clicks, target sampling rate
        '''

        if type(a_bar) == str:
            x, sr = librosa.load(a_bar, sr=sampling_rate)

        elif type(a_bar) == list:
            x, sr = a_bar, sampling_rate

        else:
            x, sr = a_bar, sampling_rate

        hop_length = 200  # samples per frame
        onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length, n_fft=2048)

        frames = range(len(onset_env))
        t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        S = librosa.stft(onset_env, hop_length=1, n_fft=512)
        fourier_tempogram = np.absolute(S)

        n0 = 0
        n1 = t.shape[0] - 1

        tmp = np.log1p(onset_env[n0:n1])  # np.log1p(x) = np.log(x+1)
        r = librosa.autocorrelate(tmp)  #

        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=400)
        tempo = librosa.beat.tempo(x, sr=sr)

        T = len(x) / float(sr)
        seconds_per_beat = 60.0 / tempo
        beat_times = np.arange(0, T, seconds_per_beat)

        clicks = librosa.clicks(beat_times, sr, length=len(x))

        with_clicks = x + clicks

        self.tempo = tempo

        return tempo, with_clicks, sampling_rate

    def tempo_compare(self, actual, user):  # Not test
        '''Finding difference to detect error'''

        # Assume they are list of tempo in different time slot
        diff = []
        position = []

        for i in range(len(actual)):
            diff.append(abs(actual[i] - user[i]))

        return np.where(diff >= 5, 1, 0)  # 1 means diff >= 4

    def spectrogram_DIY(self, list_of_filename):
        '''DIY the frequency part of spectrogram'''

        return 'Not yet finished'


#########################################################################################################


class Count_clusters:
    '''Input 1D'''

    def __init__(self, X_train_1D):
        self.what_it_is = 'Counting the number of separation notes inside a phrase'
        df = pd.DataFrame()

        tim = range(0, len(X_train_1D))
        temp_bin = []
        temp_bin_2 = []
        for i in range(len(X_train_1D)):
            if X_train_1D[i] != 0:
                temp_bin.append(tim[i])
                temp_bin_2.append(X_train_1D[i])

        df['times'] = temp_bin
        df.insert(1, 'amplitude', temp_bin_2)  # np.zeros((,len(temp_bin))
        self.df = df.reset_index(drop=True)

    def count_GMM(self, itera=10, sep=7):

        df = self.df.copy()

        # Adding iter
        ks = np.arange(1, sep)  ######## no. of separated
        length = len(ks)
        bicss = np.zeros((length,))
        aicss = np.zeros((length,))

        for i in range(0, itera):
            bics = []
            aics = []

            for k in ks:
                gmm = mixture.GaussianMixture(n_components=k, covariance_type='full', max_iter=400)
                gmm.fit(df.iloc[:, 0:1])  # 1D: gmm.fit(df.iloc[:,0:1])
                bics.append(gmm.bic(df.iloc[:, 0:1]))  # 1D: bics.append(gmm.bic(df.iloc[:,0:1]))
                aics.append(gmm.aic(df.iloc[:, 0:1]))  # # 1D: aics.append(gmm.aic(df.iloc[:,0:1]))

            bicss = bicss + np.array(bics)
            aicss = aicss + np.array(aics)

        bicss = bicss / itera
        aicss = aicss / itera

        self.gmm_clusters = ks[np.where(bicss == np.min(bicss))[0]][0]

        return self.gmm_clusters

    def count_DBSCAN(self, eps=2, min_samples=5):

        df = self.df.copy()

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(df.iloc[:, 0]).reshape(-1, 1))
        labels = clustering.labels_

        # Number of clusters in labels, ignoring noise if present.
        self.DBSCAN_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        return self.DBSCAN_clusters_, set(labels), clustering.labels_

    def count_Silhouette(self):  # Silhouette Score for K means

        df = self.df.copy()

        model = KMeans()
        # k is range of number of clusters.
        visualizer = KElbowVisualizer(model, k=(2, 15), metric='silhouette', timings=True)

        if df.iloc[:, 0:1].shape[0] - 1 <= 15:  # limit the clusters
            visualizer = KElbowVisualizer(model, k=(2, df.iloc[:, 0:1].shape[0]), metric='silhouette', timings=True)

        visualizer.fit(df.iloc[:, 0:1])
        plt.close()

        self.Silhouette_clusters = visualizer.elbow_value_

        return self.Silhouette_clusters

    def get_kmeans_score(self, data, center):  # Davies Bouldin score
        '''
        returns the kmeans score regarding Davies Bouldin for points to centers
        INPUT:
            data - the dataset you want to fit kmeans to
            center - the number of centers you want (the k value)
        OUTPUT:
            score - the Davies Bouldin score for the kmeans model fit to the data
        '''
        # instantiate kmeans
        kmeans = KMeans(n_clusters=center)
        # Then fit the model to your data using the fit method
        model = kmeans.fit_predict(data)

        # Calculate Davies Bouldin score
        score = davies_bouldin_score(data, model)

        return score

    def count_Davies(self, sep=15):  # Davies Bouldin score for K means

        df = self.df.copy()

        scores = []
        centers = list(range(2, sep))

        if df.iloc[:, 0:1].shape[0] - 1 <= max(centers):  # limit the clusters
            centers = list(range(2, df.iloc[:, 0:1].shape[0]))

        for center in centers:
            scores.append(self.get_kmeans_score(df.iloc[:, 0:1], center))

        index = np.where(np.array(scores) == min(scores))[0][0]
        self.Davies_clusters = centers[index]

        return self.Davies_clusters

    def optimalK(self, data, nrefs=3, maxClusters=15):  # Gap statistic
        """
        Calculates KMeans optimal K using Gap Statistic
        Params:
            data: ndarry of shape (n_samples, n_features)
            nrefs: number of sample reference datasets to create
            maxClusters: Maximum number of clusters to test for
        Returns: (gaps, optimalK)
        """
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})

        for gap_index, k in enumerate(range(1, maxClusters)):
            # Holder for reference dispersion results
            refDisps = np.zeros(nrefs)
            # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
            for i in range(nrefs):
                # Create new random reference set
                randomReference = np.random.random_sample(size=data.shape)

                # Fit to it
                km = KMeans(k)
                km.fit(randomReference)

                refDisp = km.inertia_
                refDisps[i] = refDisp
            # Fit cluster to original data and create dispersion
            km = KMeans(k)
            km.fit(data)

            origDisp = km.inertia_
            # Calculate gap statistic
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            # Assign this loop's gap statistic to gaps
            gaps[gap_index] = gap

            resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)

        return (gaps.argmax() + 1, resultsdf)

    def count_Gap(self):

        df = self.df.copy()

        if df.iloc[:, 0:1].shape[0] < 15:  # limit the clusters
            self.score_g, df_g = self.optimalK(df.iloc[:, 0:1], nrefs=5, maxClusters=(df.iloc[:, 0:1].shape[0] + 1))
        else:
            self.score_g, df_g = self.optimalK(df.iloc[:, 0:1], nrefs=5, maxClusters=15)

        return self.score_g

    def bagging_(self):

        gmm_clusters = self.count_GMM()
        DBSCAN_clusters_, _, _ = self.count_DBSCAN()
        Silhouette_clusters = self.count_Silhouette()
        Davies_clusters = self.count_Davies()
        score_g = self.count_Gap()

        clusters_all = [DBSCAN_clusters_, gmm_clusters, Silhouette_clusters, Davies_clusters, score_g]

        if DBSCAN_clusters_ == 1 & Davies_clusters == 14 & Silhouette_clusters == 2:  # situation for 1 cluster
            clusters_all = [DBSCAN_clusters_, gmm_clusters, DBSCAN_clusters_, score_g]

        vals, counts = np.unique(clusters_all, return_counts=True)

        index = np.argmax(counts)
        self.result = vals[index]

        return self.result, clusters_all


def classifier_(wav, sr=44100):
    audio = Audio()
    wav = np.frombuffer(wav, dtype=np.int16)
    write('temp.wav', sr, wav)
    y, sr = audio.read_audio('temp.wav', type_='librosa')
    X_train_audio, X_train_spectrogram, sr = audio.spectrogram_librosa(y, win_length=1024, n_fft=1024, sampling_rate=sr)

    for k in range(0, len(X_train_spectrogram)):
        XX_train = audio.rhythm_extract(X_train_spectrogram[k], critical_value=4, pitch=5)

    XX_train_1D = audio.converter(XX_train)
    cc = Count_clusters(XX_train_1D)

    os.remove('temp.wav')
    return cc.bagging_()[0]


# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import glob
import os
import string
import itertools
import data_validation
import threading
import codecs
import unicodedata


from tensorflow.python.platform import tf_logging as logging


# In[ ]:




# In[2]:

class DataReadAndValidate(object):
    def __init__(self, data_path='SpeechData/'):
        
        print "Base path from where data will be read: %s" % (data_path)
        self.label,self.wav = self._load_corpus(data_path)
        

    def _load_data_dir(self, data_path) :
        #print data_path
        all_train_wav_files = []
        all_train_txt_files = []
        for file in os.listdir(data_path + 'wav'):
            if file.endswith(".wav"):
                all_train_wav_files.extend([data_path + 'wav/' + file])
        for file in os.listdir(data_path + 'txt'):
            if file.endswith(".txt"):
                all_train_txt_files.extend([data_path + 'txt/' + file])
        return sorted(all_train_wav_files), sorted(all_train_txt_files)


    def _load_data_from_locations(self, location_ids) :
        all_train_wav_files = []
        all_train_txt_files = []
        for base_location in location_ids:
            data1, data2 = self._load_data_dir(base_location)
            all_train_wav_files.extend(data1)
            all_train_txt_files.extend(data2)

        return all_train_wav_files, all_train_txt_files
 


    #Load data from directory. Data stored in format for Training 
    #train/<Number>/wav/ dir contains media files
    #train/<Number>/txt/ dir contains transcription files

    #Load data from directory. Data stored in format for Testing 
    #test/<Number>/wav/ dir contains media files
    #test/<Number>/txt/ dir contains transcription files
    
    def _load_corpus(self, data_path):

        print 'Loading the speech metadata from path:', data_path
        # read meta-info
        df = pd.read_table(data_path + 'data-info.txt', usecols=['ID','Enable','Train'],
                           delimiter=',',index_col=False)

        # collect train file ids
        # make file ID
        train_file_ids = []
        test_file_ids = []
        for index, row in df.iterrows():
            if row['Enable'] == 'Y' and row['Train'] == 'Y':
                train_file_ids.extend([data_path + 'train/' + str(row['ID']) + '/'])
            if row['Enable'] == 'Y' and row['Train'] == 'N':
                train_file_ids.extend([data_path + 'test/' + str(row['ID']) + '/'])


        print 'Load data from enabled directory with training flag on :', train_file_ids
        all_train_wav_files, all_train_txt_files = self._load_data_from_locations(train_file_ids)

        #Validate the data --- Disable this of you are running again and again on same data
        all_train_wav_files, all_train_txt_files = data_validation.validate_data(all_train_wav_files, all_train_txt_files)
        all_train_wav_files = sorted(all_train_wav_files)
        all_train_txt_files = sorted(all_train_txt_files)

        return all_train_txt_files, all_train_wav_files



# In[ ]:




# In[3]:

class DictionaryBuilding(object):
    def __init__(self):
        
        #The text files will only contains english lower case alphabet with space and 0 for padding
        #creating byte to index dictionary
        self.index2byte = [0] + [ord(' ')] + list(range(ord('a'), ord('z')+1))

        self.byte2index = {}
        for i, b in enumerate(self.index2byte):
            self.byte2index[b] = i        
        self.voca_size = len(self.index2byte)
          


# In[ ]:




# In[4]:

class _FuncQueueRunner(tf.train.QueueRunner):

    def __init__(self, func, data_producer=None, queue=None, enqueue_ops=None, close_op=None,
                 cancel_op=None, queue_closed_exception_types=None,
                 queue_runner_def=None):
        # save ad-hoc function
        self.func = func
        self.data_producer = data_producer
        # call super()
        super(_FuncQueueRunner, self).__init__(queue, enqueue_ops, close_op, cancel_op,
                                               queue_closed_exception_types, queue_runner_def)

    # pylint: disable=broad-except
    def _run(self, sess, enqueue_op, coord=None):

        if coord:
            coord.register_thread(threading.current_thread())
        decremented = False
        try:
            while True:
                if coord and coord.should_stop():
                    break
                try:
                    #CUSTOM FUNCTION CALL
                    self.func(sess, enqueue_op, self.data_producer)  # call enqueue function
                    #CUSTOM FUNCTION CALL
                except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
                    # This exception indicates that a queue was closed.
                    with self._lock:
                        self._runs_per_session[sess] -= 1
                        decremented = True
                        if self._runs_per_session[sess] == 0:
                            try:
                                sess.run(self._close_op)
                            except Exception as e:
                                # Intentionally ignore errors from close_op.
                                logging.vlog(1, "Ignored exception: %s", str(e))
                        return
        except Exception as e:
            # This catches all other exceptions.
            if coord:
                coord.request_stop(e)
            else:
                logging.error("Exception in QueueRunner: %s", str(e))
                with self._lock:
                    self._exceptions_raised.append(e)
                raise
        finally:
            # Make sure we account for all terminations: normal or errors.
            if not decremented:
                with self._lock:
                    self._runs_per_session[sess] -= 1


# In[ ]:




# In[5]:



class SpeechData(object):

    def __init__(self, batch_size=16, data_path='SpeechData/'):

        #load dictionary
        dictionary = DictionaryBuilding()
        
        self.byte2index = dictionary.byte2index
        self.index2byte = dictionary.index2byte
        self.voca_size = dictionary.voca_size

        # Constants
        SPACE_TOKEN = '<space>'
        SPACE_INDEX = 0
        FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

        def text_to_char_array(original):
            r"""
            Given a Python string ``original``, remove unsupported characters, map characters
            to integers and return a numpy array representing the processed string.
            """
            # Create list of sentence's words w/spaces replaced by ''
            result = ' '.join(original.translate(None, string.punctuation).lower().split())
            result = result.replace(" '", "") # TODO: Deal with this properly
            result = result.replace("'", "")    # TODO: Deal with this properly
            result = result.replace(' ', '  ')
            result = result.split(' ')

            # Tokenize words into letters adding in SPACE_TOKEN where required
            result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])

            # Map characters into indicies
            result = np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])

            # Add result to results
            return result
        
        
        
        def _get_log_power_spectrum(wav_file):
            #All wav files are with 8k sampling rate : Taking Fourier representation: 20 ms speech to 81 feature
            sample_rate = 8000
            # load wave file with sampling rate 8000 which is already known. sr value is important
            data, sr = librosa.load(wav_file, mono=True, sr=sample_rate)

            #Short First Fourier transform - for every 20 second for 8k sampling rate= 160
            stft = librosa.stft(data, n_fft=160, hop_length=160)

            #np.abs(D[f, t]) is the magnitude of frequency bin f at frame t. Not taking the phase portion of it
            amplitude = np.abs(stft)

            #Compute dB relative to median power
            log_power_spectrogram = librosa.power_to_db(amplitude**2, ref=np.median)
            
            return log_power_spectrogram

        
        def _load_power_spectrum(src_list):
            txt_file, wav_file = src_list  # label, wave_file

            #decode string to integer ------ This could be done without dictionary also --- Need to test the difference
            # remove punctuation, to lower, clean white space 
            #sentence = ' '.join(open(txt_file).read().translate(None, string.punctuation).lower().split())
            #print 'Sentence:', sentence
            #lab = np.asarray([self.byte2index[ord(ch)] for ch in sentence])
            label = ''

            with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
                label = unicodedata.normalize("NFKD", open_txt_file.read()).encode("ascii", "ignore")
                label = text_to_char_array(label)
            label_len = len(label)

            feature = _get_log_power_spectrum(wav_file)
            feature_len = np.size(feature, 1)

            # return result
            return label, label_len, feature, feature_len

        
        # enqueue function
        def enqueue_func(sess, enqueue_op, data_producer):
            # read data from source queue
            
            label_data_file_pair = sess.run(data_producer)
          
            train_label, train_label_length, train_wave_file, train_wave_file_len = _load_power_spectrum(label_data_file_pair)

            sess.run(enqueue_op, feed_dict={label_input:train_label, label_input_length:train_label_length, feature_input:train_wave_file, feature_input_length:train_wave_file_len})
                    
                            

        # load corpus
        data_reader = DataReadAndValidate(data_path)       
        labels, wave_files = data_reader.label, data_reader.wav
        
        # calc total batch count
        self.num_batch = len(labels) // batch_size


        
        # to constant tensor
        train_labels = tf.convert_to_tensor(labels, dtype=tf.string)
        train_wave_files = tf.convert_to_tensor(wave_files, dtype=tf.string)

        data_producer = tf.train.slice_input_producer([train_labels, train_wave_files], shuffle=True)

        
        
        
        
        number_of_threads = 3        
        # 81 == Number of rows cols are variable length for speech
        # Features are [81,None] vectors of floats
        feature_input = tf.placeholder(tf.float32, shape=[81,None])
        feature_input_length = tf.placeholder(tf.int32, shape=[])
        
        # Labels are integers of variable length.
        label_input = tf.placeholder(tf.int32, shape=[None])
        label_input_length = tf.placeholder(tf.int32, shape=[])        
        
        
        padding_q = tf.PaddingFIFOQueue(16, [tf.int32, tf.int32, tf.float32, tf.int32], shapes=[[None],[],[81,None],[]])
        enqueue_op = padding_q.enqueue([label_input, label_input_length, feature_input, feature_input_length])

        runner = _FuncQueueRunner(enqueue_func, data_producer, padding_q, [enqueue_op] * number_of_threads)
        
        # register to global collection
        tf.train.add_queue_runner(runner)
      
    
        self.labels, self.labels_length, self.features, self.features_length = padding_q.dequeue_many(batch_size)
    
    
        # print info
        logging.vlog(0, "SppechData corpus loaded.(total data=%d, total batch=%d)", len(labels), self.num_batch)



    def print_index(self, indices):
        # transform label index to character
        for i, index in enumerate(indices):
            str_ = ''
            for ch in index:
                if ch > 0:
                    str_ += unichr(self.index2byte[ch])
                elif ch == 0:  # <EOS>
                    break
            print str_


# In[6]:


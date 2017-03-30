
# coding: utf-8

# In[ ]:

#imports declaration
import os
import sys
import subprocess
from subprocess import Popen,PIPE
from itertools import groupby
from collections import namedtuple
import re
import string

# In[ ]:

#Validations
#
#Every wave file should have it's txt file - This will remove not used txt file and wav files with no txt file
#Transcription should not be empty or in limit of min max sentence after removing/transforming feature
#Transcription should not contains numeric value
#Size of wav file should not be less than 1 second. (Example: with 1000 neuron we need 20*1000 ms length file = 20 sec)
#There should be at least 0.5 sec silence in the starting and 0.5 sec silence in the end of the audio file. - Code 
#Incomplete words from the ends of the audio files should be discarded. - Code


# In[ ]:

#select wav files whole transcription is available
#We need the files which will have voicelogs and its associated files.
#Also prints files which do not have transcription for voicelog or remove transcrioption which do not have voicelog


max_length=100
min_length=3
min_audio_length=0

def  mapping_wav_to_txt_files(wav_files, txt_files):
    
    txt_files_filtered = filter( lambda file_name : file_name.replace('/txt','/wav') in wav_files , txt_files)
    wav_files_filtered = filter( lambda file_name : file_name.replace('/wav','/txt') in txt_files , wav_files)

    # Wav files with no transcript
#    wav_files_with_no_txt = [filter(lambda x: x in wav_files_filtered, sublist) for sublist in wav_files]
#    print 'Wav files with no transcript:'
#    print wav_files_with_no_txt
#    #Txt files with no wav files
#    txt_files_with_no_wav = [filter(lambda x: x in txt_files_filtered, sublist) for sublist in txt_files]
#    print 'Txt files with no wav files:'
#    print txt_files_with_no_wav
    #print len(txt_files_filtered)
    return wav_files_filtered, txt_files_filtered


# In[ ]:

def remove_files_with_invalid_transcription(valid_files, max_length, min_length):
    wrong_length_files = []
    correct_files = []
    non_alphabet_files = []
    for file_name in valid_files: 
        words = open(file_name+'.txt').read().translate(None, string.punctuation).lower().split()
        
        if (''.join(words)).isalpha():
            if len(words) < min_length or len(words) > max_length:
                wrong_length_files.append(file_name)
	
            else:
                correct_files.append(file_name)
        else:
            non_alphabet_files.append(file_name)
    print 'Length CHeck Failed: '
    print wrong_length_files

    print 'Wrong transcription. contains numeric or special character'
    print non_alphabet_files

    return correct_files



# In[ ]:

def media_validation_on_files(valid_check_files, min_audio_length):
    wrong_media_files = []
    valid_media_files = []
    for valid_check_file in valid_check_files:
        out,err = (subprocess.Popen(['soxi','-D',valid_check_file + '.wav'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)).communicate()
	#print err
        try :
              length = float(out)
              if length > min_audio_length :
                  valid_media_files.append(valid_check_file)
              else:
                  wrong_media_files.append(valid_check_file)
        except ValueError:
             wrong_media_files.append(valid_check_file) 
    print 'Wrong media files:'
    print wrong_media_files
    return valid_media_files




# In[ ]:

def validate_data(wav_files, txt_files):

    global max_length
    global min_length
    global min_audio_length
    txt_files = map( lambda file_name : file_name.replace('.txt','') , txt_files)
    wav_files = map( lambda file_name : file_name.replace('.wav','') , wav_files)
    print len(txt_files)
    wav_files=media_validation_on_files(wav_files, min_audio_length)
    print "wav file length after validation"
    print len(wav_files)


    original_wav_files= wav_files
    original_txt_files= txt_files
    #print len(original_wav_files)
    #print len(original_txt_files)
    txt_files= remove_files_with_invalid_transcription(txt_files, max_length, min_length)
    wav_files, txt_files= mapping_wav_to_txt_files(wav_files, txt_files)
    #print len(wav_files)
    
    print  set (original_wav_files) - set(wav_files)
    print  set (original_txt_files) - set(txt_files)

    txt_files = map( lambda file_name : file_name + '.txt' , txt_files)
    wav_files = map( lambda file_name : file_name + '.wav' , wav_files)
    print len(txt_files)
    print len(wav_files)
    return wav_files, txt_files
    
    
    


# In[ ]:




# In[ ]:





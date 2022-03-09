import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random

# Train Val Test Split beforehand

def dataset_split(audio_code_list, test_size=0.2):
  '''
  Inputs
  audio_code_list: list of tuples of audio_data and code

  Outputs
  train, validation and test set as list of tuples of audio_data and code
  '''
  audio_data=[]
  codes= []

  for ad, c in audio_code_list:
    audio_data.append(ad)
    codes.append(c)

  X_train, X_test, y_train, y_test = train_test_split(audio_data,codes,test_size=test_size)
  X_train, X_val, y_train, y_val   = train_test_split(X_train,y_train,test_size=test_size)

  train_set=[]
  val_set  =[]
  test_set =[]

  for ad,c in zip(X_train,y_train):
      train_set.append((ad,c))

  for ad,c in zip(X_val,y_val):
      val_set.append((ad,c))

  for ad,c in zip(X_test,y_test):
      test_set.append((ad,c))

  return train_set,val_set, test_set

def enhanced_dataset_split(input_list,df, test_size=0.2, threshold=0.5, target='family'):
  '''
  Inputs
  input_list: list of tuples of audio_data and code
  df: DataFrame with specific info, defined with get_dataset()
  test_size: test_size required by sklearn train_test_split function
  threshold: determines the class distribution in the train set

  Outputs
  train, validation and test set as list of tuples of audio_data and code
  '''
  # it = 0

  if target == 'family':
    condition = False
    while condition == False:
      train_set,val_set, test_set = dataset_split(input_list, test_size=test_size) # train val test split

      X_train_temp=[]
      y_train_temp=[]

      for x,y in train_set:
        X_train_temp.append(x)
        y_train_temp.append(y)

      df_total= pd.DataFrame(df[['family_code']].value_counts()*threshold).rename(columns={0:'total_count'})
      df_temp=pd.DataFrame(pd.DataFrame(y_train_temp).value_counts()).rename(columns={0:'count'})
      df_temp['total_count']=df_total['total_count']
      df_temp['comparison']=np.where(df_temp['count'] >= df_temp['total_count'], 'True', 'False')

      if df_temp[df_temp['comparison'] == 'True'].count()['comparison'] == df_temp.shape[0]:
        condition = True

  elif target == 'species':
    condition = False
    while condition == False:
      train_set,val_set, test_set = dataset_split(input_list, test_size=test_size) # train val test split

      X_train_temp=[]
      y_train_temp=[]

      for x,y in train_set:
        X_train_temp.append(x)
        y_train_temp.append(y)

      df_total= pd.DataFrame(df[['species_code']].value_counts()*threshold).rename(columns={0:'total_count'})
      df_temp=pd.DataFrame(pd.DataFrame(y_train_temp).value_counts()).rename(columns={0:'count'})
      df_temp['total_count']=df_total['total_count']
      df_temp['comparison']=np.where(df_temp['count'] >= df_temp['total_count'], 'True', 'False')

      if df_temp[df_temp['comparison'] == 'True'].count()['comparison'] == df_temp.shape[0]:
        condition = True
    # it += 1
  return train_set,val_set, test_set

# Split audio files that are above and below target_time

def split_above_below(audio_code_list, target_time,sr):
  '''
  Inputs
  audio_code_list: list of tuples of audio_data and code
  target_time    : duration in seconds wanted for the audio_data
  sr             : sampling rate

  Outputs
  below: list of tuples of audio_data and code below or equal to target_time
  above: list of tuples of audio_data and code above target_time

  '''
  below = []
  above = []

  for ad, c in audio_code_list:
    if (len(ad) / sr) < target_time:
      below.append((ad,c))
    else:
      above.append((ad,c))

  return below,above

# Preprocessing samples that are above target_time

def train_split_above_samples(above,over_r,under_r, target_time, sr):
  '''
  Inputs
  above      : list of tuples of audio_data and code above target_time
  over_r     : list of code names of over_represented classes
  under_r    : list of code names of under_represented classes
  target_time: duration in seconds wanted for the audio_data
  sr         : sampling rate

  Ouputs
  above_record_samples: list of tuples of audio_data and code of duration target_time,
                  with different preprocessing based on class representation
  '''
  under_r_samples = []
  over_r_samples  = []

  for audio_data, code in above:
      if code in under_r: # under-represented classes
        # cut in 5s consecutive slices 3 times at different intervals (Victoria)
      # for complete 5s extracts
        nb_split_samples = audio_data.size // (target_time*sr)

        for i in range(0,nb_split_samples):
          sound_1 = audio_data[i*(target_time*sr): (i+1)*(target_time*sr)]
          under_r_samples.append((sound_1,code))

          random_shift = random.randint(0,((target_time*sr)/3)*2)
          if len(audio_data) > ((i+1)*(target_time*sr))+random_shift:
            sound_2 = audio_data[((i*(target_time*sr))+random_shift): (((i+1)*(target_time*sr))+random_shift)]
            under_r_samples.append((sound_2,code))

          random_shift = random.randint(0,((target_time*sr)/3)*2)
          if len(audio_data) > ((i+1)*(target_time*sr))+random_shift:
            sound_3 = audio_data[((i*(target_time*sr))+random_shift): (((i+1)*(target_time*sr))+random_shift)]
            under_r_samples.append((sound_3,code))

      else:  # over-represented classes
          nb_split_samples = len(audio_data) // (target_time*sr) # cut in 5s consecutive slices + pad last one if >= target_time - 1

          for i in range(0,nb_split_samples):
            sound_i = audio_data[i*(target_time*sr): (i+1)*(target_time*sr)]
            under_r_samples.append((sound_i, code))

            if (len(audio_data) % (target_time*sr)) / sr >= target_time - 1 :
              len_to_pad=(sr*target_time) - len(audio_data[(nb_split_samples) * (target_time * sr):])
              a = random.randint(0, len_to_pad)
              b = len_to_pad - a
              audio = np.pad(audio_data[(nb_split_samples)*(target_time*sr):], (a,b), "constant")
              under_r_samples.append((audio, code))

  above_record_samples  = under_r_samples + over_r_samples

  return above_record_samples

# Preprocessing samples that are below target_time

def train_split_below_samples(below,over_r,under_r, target_time, sr):
  '''
  Inputs
  below      : list of tuples of audio_data and code below or equal to target_time
  over_r     : list of code names of over_represented classes
  under_r    : list of code names of under_represented classes
  target_time: duration in seconds wanted for the audio_data
  sr         : sampling rate

  Ouputs
  below_record_samples: list of tuples of audio_data padded randomly and code
  '''
  under_r_samples = []
  over_r_samples  = []

  for audio_data, code in below:
    if code in under_r: # under-represented classes

      for i in range(3):  # 3 random pads per sample
          len_to_pad=(sr*target_time) - len(audio_data)
          a = random.randint(0, len_to_pad)
          b = len_to_pad - a
          audio = np.pad(audio_data, (a,b), "constant")
          under_r_samples.append((audio,code))

    else: # over-represented classes
         # pad randomly
        len_to_pad=(sr*target_time) - len(audio_data)
        a = random.randint(0, len_to_pad)
        b = len_to_pad - a
        audio = np.pad(audio_data, (a,b), "constant")
        over_r_samples.append((audio,code))

  below_record_samples  = under_r_samples + over_r_samples

  return below_record_samples

# Compile final Train set

def final_set(above_record_samples,below_record_samples):
  '''
  Inputs
  above_record_samples: list of tuples of audio_data padded issued from ***_split_above_samples functions
  below_record_samples: list of tuples of audio_data padded issued from ***_split_below_samples functions

  Outputs
  preprocessed train set
  '''
  return above_record_samples+below_record_samples

# Split audio files that are above and below target_time for Val and Test sets

def split_above_below(audio_code_list, target_time,sr):
  '''
  Inputs
  audio_code_list: list of tuples of audio_data and code
  target_time    : duration in seconds wanted for the audio_data
  sr             : sampling rate

  Outputs
  below: list of tuples of audio_data and code below or equal to target_time
  above: list of tuples of audio_data and code above target_time

  '''
  below = []
  above = []

  for ad, c in audio_code_list:
    if (len(ad) / sr) <= target_time:
      below.append((ad,c))
    else:
      above.append((ad,c))

  return below,above

# Preprocessing Val and Test samples that are above target_time

def val_test_split_above_samples(above, target_time, sr):
  '''
  Inputs
  above      : list of tuples of audio_data and code above target_time
  target_time: duration in seconds wanted for the audio_data
  sr         : sampling rate

  Ouputs
  above_record_samples: list of tuples of audio_data and code of duration target_time

  '''
  above_samples=[]

  # cut in 5s consecutive slices + pad last one if >= target_time - 1
  for audio_data, code in above:
    nb_split_samples = len(audio_data) // (target_time*sr)

    for i in range(0,nb_split_samples):
      sound_i = audio_data[i*(target_time*sr): (i+1)*(target_time*sr)]
      above_samples.append((sound_i, code))

      if (len(audio_data) % (target_time*sr)) / sr >= target_time - 1 :
        len_to_pad=(sr*target_time) - len(audio_data[(nb_split_samples) * (target_time * sr):])
        a = random.randint(0, len_to_pad)
        b = len_to_pad - a
        audio = np.pad(audio_data[(nb_split_samples)*(target_time*sr):], (a,b), "constant")
        above_samples.append((audio, code))

  return above_samples

  # Preprocessing Val and Test samples that are below target_time

def val_test_split_below_samples(below, target_time, sr):
  '''
  Inputs
  below      : list of tuples of audio_data and code below or equal to target_time
  target_time: duration in seconds wanted for the audio_data
  sr         : sampling rate

  Ouputs
  below_record_samples: list of tuples of audio_data padded randomly and code
  '''
  below_record_samples=[]

  for audio_data, code in below:
    if (len(audio_data) / sr) < target_time:
      len_to_pad=(sr*target_time) - len(audio_data)
      a = random.randint(0, len_to_pad)
      b = len_to_pad - a
      audio = np.pad(audio_data, (a,b), "constant")
      below_record_samples.append((audio,code))
    else:
      below_record_samples.append((audio_data,code))

  return below_record_samples

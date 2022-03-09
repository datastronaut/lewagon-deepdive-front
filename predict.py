from utils.dataset import get_arrays_from_audio
from utils.data_preproc import val_test_split_below_samples, val_test_split_above_samples
import numpy as np

def predict_class(audio_data, model):
    TARGET_TIME = 5
    SR=44_100 #sampling rate in Hz
    species_code='dummy'

    tuple_pred = [(audio_data, species_code)]

    if len(audio_data)/SR < TARGET_TIME:
        tuple_pred = val_test_split_below_samples(tuple_pred,TARGET_TIME,SR)
    else:
        tuple_pred = val_test_split_above_samples(tuple_pred,TARGET_TIME,SR)

    tuple_pred = get_arrays_from_audio(tuple_pred)
    pred=0

    for i in range(len(tuple_pred)):
        x= tuple_pred[i][0].reshape((1,) + tuple_pred[i][0].shape)
        pred += model.predict(np.expand_dims(x,-1))

    pred = pred/len(tuple_pred)
    print('pred = ',pred)

    ind = np.argpartition(pred[0], -3)[-3:]
    print('ind = ',ind)

    proba = [round(p*100,2) for p in pred[0][ind]]
    print('proba = ',proba)

    return {key: value for key, value in zip(ind, proba)}

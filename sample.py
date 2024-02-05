from ispa import utils
from ispa.acoustics import run_inference as ispa_a_run_inference
from ispa.features import FeatureBasedISPAPredictor

waveform, sr = utils.load_waveform('1-38560-A-14.wav')
ispa_results = ispa_a_run_inference(waveform, sr)
print(ispa_results['text'])

ispa_f_predictor = FeatureBasedISPAPredictor(
    feature_type='mfcc',
    kmeans_model='models/kmeans.mfcc.pkl',
    phoneme_map='models/c2p.mfcc.json')
print(ispa_f_predictor.predict(waveform, variation='raw'))
print(ispa_f_predictor.predict(waveform, variation='seg'))
print(ispa_f_predictor.predict(waveform, variation='phn'))

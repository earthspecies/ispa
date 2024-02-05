from ispa import utils
from ispa.acoustics import run_inference as ispa_a_run_inference
from ispa.features import FeatureBasedISPAPredictor

waveform, sr = utils.load_waveform('1-38560-A-14.wav')
ispa_results = ispa_a_run_inference(waveform, sr)
print("ISP-A results:")
print(ispa_results['text'])
print()

ispa_f_predictor = FeatureBasedISPAPredictor(
    feature_type='mfcc',
    kmeans_model='models/kmeans.mfcc.pkl',
    phoneme_map='models/c2p.mfcc.json')
print("ISP-F results (with MFCC):")
print("(raw):", ispa_f_predictor.predict(waveform, variation='raw'))
print("(seg):", ispa_f_predictor.predict(waveform, variation='seg'))
print("(phn):", ispa_f_predictor.predict(waveform, variation='phn'))
print()

ispa_f_predictor = FeatureBasedISPAPredictor(
    feature_type='aves',
    kmeans_model='models/kmeans.aves.pkl',
    phoneme_map='models/c2p.aves.json',
    aves_config_path='models/aves-base-bio.torchaudio.model_config.json',
    aves_model_path='models/aves-base-bio.torchaudio.pt')
print("ISP-F results (with AVES):")
print("(raw):", ispa_f_predictor.predict(waveform, variation='raw'))
print("(seg):", ispa_f_predictor.predict(waveform, variation='seg'))
print("(phn):", ispa_f_predictor.predict(waveform, variation='phn'))
print()

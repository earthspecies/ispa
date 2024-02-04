from ispa import utils
from ispa.acoustics import run_inference

waveform, sr = utils.load_waveform('1-38560-A-14.wav')
ispa_results = run_inference(waveform, sr)
print(ispa_results['text'])

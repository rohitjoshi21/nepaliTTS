#!/usr/bin/env python
# coding: utf-8

# # Nepali Text-to-Speech with Tacotron2 and Waveglow
# 

# In[1]:





# In[1]:


#@title Clone the REPOS
import os
from os.path import exists, join, basename, splitext
git_repo_url = 'https://github.com/NVIDIA/tacotron2.git'
project_name = splitext(basename(git_repo_url))[0]
print(project_name)

import sys
sys.path.append('hifi-gan')
sys.path.append(project_name)
import time
import matplotlib
import matplotlib.pylab as plt

import numpy as np
import torch
import json
from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from text import text_to_sequence
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator


# In[2]:


#@title Creating Hyperparameters

from text import symbols

class HParams:
    def __init__(self) -> None:
        self.epochs=500
        self.iters_per_checkpoint=1000
        self.seed=1234
        self.dynamic_loss_scaling=True
        self.fp16_run=False
        self.distributed_run=False
        self.dist_backend="nccl"
        self.dist_url="tcp://localhost:54321"
        self.cudnn_enabled=True
        self.cudnn_benchmark=False
        self.ignore_layers=['embedding.weight']

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk=False
        self.training_files='./filelists/train_list.txt'
        self.validation_files='./filelists/val_list.txt'
        self.text_cleaners=['transliteration_cleaners']

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value=32768.0
        self.sampling_rate=22050
        self.filter_length=1024
        self.hop_length=256
        self.win_length=1024
        self.n_mel_channels=80
        self.mel_fmin=0.0
        self.mel_fmax=8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols=len(symbols)
        self.symbols_embedding_dim=512

        # Encoder parameters
        self.encoder_kernel_size=5
        self.encoder_n_convolutions=3
        self.encoder_embedding_dim=512

        # Decoder parameters
        self.n_frames_per_step=1  # currently only 1 is supported
        self.decoder_rnn_dim=1024
        self.prenet_dim=256
        self.max_decoder_steps=1000
        self.gate_threshold=0.5
        self.p_attention_dropout=0.1
        self.p_decoder_dropout=0.1

        # Attention parameters
        self.attention_rnn_dim=1024
        self.attention_dim=128

        # Location Layer parameters
        self.attention_location_n_filters=32
        self.attention_location_kernel_size=31

        # Mel-post processing network parameters
        self.postnet_embedding_dim=512
        self.postnet_kernel_size=5
        self.postnet_n_convolutions=5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate=False
        self.learning_rate=1e-3
        self.weight_decay=1e-6
        self.grad_clip_thresh=1.0
        self.batch_size=8
        self.mask_padding=True  # set model's padded outputs to padded values

hparams = HParams()


# In[3]:


#@title Load Tacotron2 &  HiFi-GAN

#@markdown Config:

#Universal HiFi-GAN (has some robotic noise): 1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW
Tacotron2_Model = 'Shruti_22kHz/archive.zip'#@param {type:"string"}
TACOTRON2_ID = Tacotron2_Model
HIFIGAN_ID = "1qpgI41wNXFcH-iKq1Y42JlBC9j0je8PW"


graph_width = 900
graph_height = 360
def plot_data(data, figsize=(int(graph_width/100), int(graph_height/100))):
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                    interpolation='none', cmap='inferno')
    fig.canvas.draw()
    plt.show()

# !gdown --id '1E12g_sREdcH5vuZb44EZYX8JjGWQ9rRp'
thisdict = {}
for line in reversed((open('merged.dict.txt', "r").read()).splitlines()):
    thisdict[(line.split(" ",1))[0]] = (line.split(" ",1))[1].strip()

def ARPA(text, punctuation=r"!?,।.;", EOS_Token=True):
    out = ''
    for word_ in text.split(" "):
        word=word_; end_chars = ''
        while any(elem in word for elem in punctuation) and len(word) > 1:
            if word[-1] in punctuation: end_chars = word[-1] + end_chars; word = word[:-1]
            else: break
        try:
            word_arpa = thisdict[word.upper()]
            word = "{" + str(word_arpa) + "}"
        except KeyError: pass
        out = (out + " " + word + end_chars).strip()
    if EOS_Token and out[-1] != ";": out += ";"
    return out

def get_hifigan(MODEL_ID):
    # Download HiFi-GAN
    hifigan_pretrained_model = 'hifimodel'
    # gdown.download("https://drive.google.com/uc?id="+MODEL_ID, hifigan_pretrained_model, quiet=False)
    if not exists(hifigan_pretrained_model):
        raise Exception("HiFI-GAN model failed to download!")

    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", "config_v1.json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cpu"))
    state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cpu"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan, h

def has_MMI(STATE_DICT):
    return any(True for x in STATE_DICT.keys() if "mi." in x)

def get_Tactron2(MODEL_ID):
    tacotron2_pretrained_model = TACOTRON2_ID
    if not exists(tacotron2_pretrained_model):
        raise Exception("Tacotron2 model failed to download!")
    # Load Tacotron2 and Config
    hparams.sampling_rate = 22050
    hparams.max_decoder_steps = 3000 # Max Duration
    hparams.gate_threshold = 0.25 # Model must be 25% sure the clip is over before ending generation
    model = Tacotron2(hparams)
    state_dict = torch.load(tacotron2_pretrained_model,map_location=torch.device('cpu'))['state_dict']
    if has_MMI(state_dict):
        raise Exception("ERROR: This notebook does not currently support MMI models.")
    model.load_state_dict(state_dict)
    _ = model.cpu().eval().half()
    return model



# In[4]:


#@title Infer Audio Method
def end_to_end_infer(text, pronounciation_dictionary, show_graphs):
    for i in [x for x in text.split("\n") if len(x)]:
        if not pronounciation_dictionary:
            if i[-1] != ";": i=i+";" 
        else: i = ARPA(i)
        with torch.no_grad(): # save VRAM by not including gradients
            sequence = np.array(text_to_sequence(i, ['transliteration_cleaners']))[None, :]
            sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cpu().long()
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
            if show_graphs:
                plot_data((mel_outputs_postnet.float().data.cpu().numpy()[0],
                        alignments.float().data.cpu().numpy()[0].T))
            y_g_hat = hifigan(mel_outputs_postnet.float())
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            print("")
            ipd.display(ipd.Audio(audio.cpu().numpy().astype("int16"), rate=hparams.sampling_rate))


# In[5]:


#@title Get Modal & HiFIGAN
hifigan, h = get_hifigan(HIFIGAN_ID)
model = get_Tactron2(TACOTRON2_ID)
previous_tt2_id = TACOTRON2_ID


# In[6]:


#@title Configuration before Infer

pronounciation_dictionary = False #@param {type:"boolean"}
# disables automatic ARPAbet conversion, useful for inputting your own ARPAbet pronounciations or just for testing
show_graphs = False #@param {type:"boolean"}
max_duration = 25 #this does nothing
model.decoder.max_decoder_steps = 10000 #@param {type:"integer"}
stop_threshold = 0.324 #@param {type:"number"}
model.decoder.gate_threshold = stop_threshold


# In[ ]:


#@title Synthesize a text
print(f"Current Config:\npronounciation_dictionary: {pronounciation_dictionary}\nshow_graphs: {show_graphs}\nmax_duration (in seconds): {max_duration}\nstop_threshold: {stop_threshold}\n\n")

time.sleep(1)
print("Enter/Paste your text.")
contents = []
while True:
    try:
        print("-"*50)
        line = input()
        if line == "":
            continue
        end_to_end_infer(line, pronounciation_dictionary, show_graphs)
    except EOFError:
        break
    except KeyboardInterrupt:
        print("Stopping...")
        break







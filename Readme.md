# **Fake My Voice**

**“Clone any voice. Speak in anyone’s tone.”**

---

## **Table of Contents**

1. About the Project  
- Aim  
- System overview  
- Tech stack   
2. Architecture  
- Workflow  
- Dataset  
3. Results & Demos  
4. File Structure  
5. Challenges Faced  
6. Contributors  
7. Mentors  
8. Acknowledgements & References

---

##  **About the Project**

- Fake My Voice is a deep learning–based project focused on voice cloning and speaker-conditioned speech synthesis.  
- The system can generate realistic speech that sounds like a specific person using only a few seconds of their audio.  
- It combines speaker embeddings, sequence-to-sequence speech synthesis, and neural vocoding for high-quality output.  
- Built as part of an exploration into multi-speaker text-to-speech (TTS) models.

---

##  **Aim**

- To build a multi-speaker voice cloning system capable of replicating a target speaker’s tone, accent, and style.  
- To implement and understand the working of Tacotron2, GE2E Speaker Encoder, and WaveGlow in a unified pipeline.  
- To produce natural, human-like speech from text with minimal data per speaker.

---

##  **System Overview**

1. ## Extract speaker embedding from reference audio.

2. ## Use embedding \+ text to generate mel-spectrogram.

3. ## Feed mel-spectrogram into vocoder for waveform generation.

4. ## Output is speech mimicking the original speaker.

---

## Tech Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Language** | Python |
| **Frameworks** | PyTorch, NumPy, Librosa |
| **Audio Tools** | Torchaudio, SoundFile, Matplotlib |
| **Models Used** | GE2E Speaker Encoder, Tacotron2, WaveGlow |
| **Visualization** | TensorBoard |

## ---

## **Architecture**

![architecture][images/architecture.png]

## Flow explained: 

###  1\. **Speaker Encoder (GE2E)**

- Learns to extract a fixed-dimensional embedding (256-D) that captures the unique vocal identity of a speaker.  
- Trained using the Generalized End-to-End (GE2E) loss, which encourages embeddings from the same speaker to cluster closely while separating different speakers in the embedding space.  
- Input: A few seconds of raw audio from the target speaker.  
- Output: A speaker embedding vector that serves as a conditioning input for the speech synthesizer.  
  ---

  ###  2**. Speech Synthesizer (Tacotron2)**

- Converts text input into a mel-spectrogram conditioned on the extracted speaker embedding.  
- Tacotron2 is a sequence-to-sequence model with attention, and it consists of three main submodules:

  #### a. **Encoder**

- Processes text input (after phoneme or character embedding).  
- Uses a stack of convolutional and recurrent layers to capture linguistic and contextual information from the text sequence.  
- Outputs a sequence of high-level feature representations.

  #### b. **Location-Sensitive Attention**

- Aligns encoder outputs with decoder time steps.  
- Ensures smooth, monotonic progression between text and generated speech frames.  
- Prevents alignment errors such as skipped or repeated words.

  #### c. **Decoder**

- Autoregressively generates mel-spectrogram frames one step at a time.  
- Uses both the attended context (from attention mechanism) and previous outputs to predict the next mel frame.  
- The model is trained with L1 loss.  
- Output: A high-resolution mel-spectrogram representing the audio features corresponding to the given text and speaker.  
  ---

  ###  3\. **Vocoder (WaveGlow)**

- Converts the mel-spectrogram into a time-domain waveform (audible audio).  
- Input: Mel-spectrogram from Tacotron2.  
- Output: Final waveform audio in the cloned speaker’s voice.  
  ---

  ###  **End-to-End Process**

1. Input:

   - Short audio samples of the target speaker.  
   - Text to be spoken.

2. Processing Steps:

   - Speaker Encoder extracts the speaker embedding.  
   - Tacotron2 generates a mel-spectrogram conditioned on this embedding and the input text.  
   - Vocoder converts the mel-spectrogram into a natural-sounding waveform.

3. Output:

   - Speech audio that mimics the target speaker’s tone, pitch, and speaking style, producing a highly realistic cloned voice.

## ---

##  **Workflow**

1. ## Data Preparation:

   - ## Load and preprocess audio samples (trim, normalize, resample to 22.05 kHz).

   - ## Generate mel-spectrograms.

2. ## Speaker Encoder Training:

   - ## Use GE2E loss to train on multi-speaker dataset (VCTK).

3. ## Speech Synthesis (Tacotron2):

   - ## Train on text \+ mel pairs conditioned with embeddings.

4. ## Vocoder (WaveGlow):

   - ## Generate waveform from mel-spectrogram.

5. ## Inference:

   - ## Provide text and reference audio → get cloned speech output.

## ---

## **Dataset**

* ## Dataset Used: 

- LJSpeech \- for tacotron2  
- Voxceleb \- for speaker embeddings   
- VCTK Corpus \- for tacotron2+speaker embeddings merged model 

* ## Details:

  ## For LJSpeech: 
  - \~13k samples, 1 single speaker. 

  ## For Vox celeb: 

  - \~5k samples, 40 English speakers.

  ## For VCTK Corpus: 

  - 44 hours of speech, 109 English speakers.

  - Sampling rate converted from 48 kHz → 22.05 kHz.

  - Each utterance normalized and truncated to ≤ 4 s..

---

##  **Results & Demos**

Tacotron2 \-

\- Smooth melspectrogram  after \~60 epochs with stop token prediction.

Text : Printing, in the only sense with which we are at present concerned

- Expected mel:

![Expected mel](images/target_mel.png)

- Predicted mel:

![mel predicted](images/mel_64.png)

- Predicted Frames:

![output frames](images/output_frames.png)

- Speaker embeddings \- 

\- Achieved stable embedding separation across 40+ speakers.

\- t-SNE plots show distinct clustering per speaker identity.

![speaker identites](images/embeddings.jpg)

**Losses and Accuracy:**

- **Mel-Spectrogram Prediction:** Uses Mean Squared Error (MSE) Loss to measure the difference between predicted and target mel-spectrograms, ensuring accurate reconstruction of acoustic features.

- **Stop Token Prediction:** Uses Binary Cross-Entropy (BCE) Loss to determine when the model should stop generating frames, preventing unnecessary or incomplete audio outputs.  
![loss graph plot](images/loss.png)

---

## **File Structure**

Fake\_My\_Voice/  
├── Multi-Speaker-TTS/          \# Contains code and models for multi-speaker text-to-speech synthesis  
├── SingleSpeaker\_TTS/          \# Contains code and models for single-speaker text-to-speech synthesis  
├── Speaker\_Embeddings/         \# Contains scripts and models for speaker embedding extraction  
├── datasets.txt                \# List of datasets used for training  
└── requirements.txt            \# Python dependencies for the project

## `└── README.md`

---

 **Challenges Faced**

* Mismatch between mel and waveform lengths during training.  
* Handling different sampling rates (48 kHz → 22.05 kHz) during fine tuning.  
* Managing GPU memory usage with Tacotron2.  
* Silent or noisy outputs due to alignment instability.  
* Training convergence dependent on embedding quality.

---

##  **Contributors**

* Aryan Doshi   
* Dhiraj Shirse  
* Nihira Neralwar

---

## **Mentors**

* Kevin Shah  
* Prasanna Kasar  
* Yash Ogale

---

##  **Acknowledgements & References**

- Community of Coders and Project X VJTI for providing this opportunity.   
- [Tacotron2 Paper (Google)](https://arxiv.org/pdf/1712.05884)  
- [GE2E Speaker Encoder (Google)](https://arxiv.org/pdf/1710.10467)  
- [WaveGlow (NVIDIA)](https://arxiv.org/pdf/1811.00002)  
- [LJSpeech Dataset](https://www.kaggle.com/datasets/dromosys/ljspeech)  
- [VoxCeleb Dataset](https://www.kaggle.com/datasets/bachng/voxceleb)  
- [VCTK Corpus Dataset](https://www.kaggle.com/datasets/pratt3000/vctk-corpus)  
- [NVIDIA Tacotron2 \+ WaveGlow PyTorch Implementation](https://github.com/NVIDIA/tacotron2)  
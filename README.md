# Emotion-Based Control System for Drone Choreography

Our system translates audio and rhythm of music into drone swarm choreography reflecting the emotion in the music. This project explores how emotional information in music can be converted into synchronized drone motion to improve drone light shows and performances.

The system consists of a software pipeline capable of extracting emotions from music and generating corresponding drone motions. The project also includes the design of evaluation metrics and potential real-world testing with drones in a real-world setting. A user study would be designed and conducted to evaluate the system’s ability and effectiveness in delivering precise emotions to audiences. This work has potential applications in artistic drone shows and live concerts.

# Why This Matters

Drone choreography today is expensive, manual and hard to scale.
Our system automates this process by:
- Extracting emotional progression from music
- Mapping it to drone swarm motion
- Enhancing performances with synchronized lighting
This allows us to produce a result that is scalable, expressive and efficient.

# Key Features
- 🎧 Emotion-driven drone choreography based on music analysis  
- 💡 Dynamic LED lighting that reflects emotional changes  
- 🥁 (In Progress) Beat-based lighting synchronization  
- 🐦 Swarm-based motion using Boids principles  
- 🎼 Multi-song support for diverse performances

# Background & Research Context

While prior work [https://arxiv.org/abs/2312.01059] has linked emotions to swarm behaviors or focused on beat-aligned drone performances, few systems model how emotions evolve dynamically within a song, nor do they capture the tension within a song. We aim to replicate this emotional sensitivity in a robotic swarm context.

# System Pipeline

Audio → segmentation (verse, chorus, etc.)  
→ emotion extraction (CLAP + valence-arousal)  
→ emotion + description fusion  
→ motion generation (swarm behaviors)  
→ trajectory generation  
→ drone execution + lighting control

Audio → segment the song into verse, chorus, etc. [DONE] then chop it down further → CLAP (maybe put some controllable knobs here too) https://huggingface.co/laion/larger_clap_general→ list(timestamps, descriptions)
Audio → Valence-arousal (need to adjust)→ list(timestampls, emotion weights) [basically what we had before]

# Demo Videos
https://drive.google.com/file/d/1b4hvnm46FOH6T7PuVtn1_fpS_iIXRZNy/view?usp=share_link
https://drive.google.com/file/d/1KHLcVVItElIULGh1YBctTaW9qaRAD5L8/view?usp=sharing
https://drive.google.com/file/d/13fKHg1Tnui9uJ31wvQyziR69g2Agij96/view?usp=sharing

# Team Members & Acknowledgements
Gordon Shum - CLAP Model Lead
(Implementing CLAP audio model, emotion recognition pipeline)
Henry James - LLM Implementation Lead
(Implementing emotion mapping and communication interface)
Lydia Brown - Boids Simulation/Visualization Lead
(Implementing drone simulation and boids algorithm)
Dr. Nicole Fronda - Project Mentor
Dr. Houssam Abbas - OSU Assistant Professor in EECS


##json

All our data is kept in the ```/json``` folder


## Valence-arousal to Emotion
We uses the Russuell complex to classify our emotion from the music. The va score is calculated using essentia library which does not
work on Windows. If you need to run ```predict_valence_arousal_values``` function in ```valence_arousal.py```, you would have to
install essentia on Linux (I use WSL) or Macos (unless you are ok with building from source). 

## Segmentation
the structural segmentation is done using the models from the following github repo. Detailed instruction could be found in https://github.com/justinsalamon/musicseg_deepemb

The beat tracking uses librosa's beat track function

```t
@inproceedings{Salamon:Segmentation:ISMIR:2021,
	Author = {J. Salamon and O. Nieto and N.J. Bryan},
	Booktitle = {Proc.~22nd International Conference on Music Information Retrieval (ISMIR)},
	Month = {Nov.},
	Title = {Deep Embeddings and Section Fusion Improve Music Segmentation},
	Year = {2021}}
```


## Sources

https://drive.google.com/drive/folders/1biQv9RM5Vp0RZNin3hbTd6s_QeQOvaPC?usp=drive_link


## Trajectories

use https://github.com/whoenig/uav_trajectories to convert xyz positions to Poly4D trajectory csv

build the repo following their instruction and in the build folder, run something like
```python3 ../scripts/generate_trajectory.py ../../../tmp_0.csv ../../../traj1.csv```


## How to build osqp

```
cd visuals
cd osqp
git clone --recurse-submodules https://github.com/osqp/osqp.git
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DOSQP_BUILD_TESTS=OFF
cmake --build . -j
```

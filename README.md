# Musical Drone

This project explores how emotional information embedded in music signals can be translated into expressive choreographic motion for UAV drone swarms. The goal is to enhance drone light shows and music-based performances with generated drone motion based on the emotional progression of a musical piece. The system will consist of a software pipeline capable of extracting emotions from music and generating corresponding drone motions. The project also includes the design of evaluation metrics and potential real-world testing with drones in a real-world setting. A user study would be designed and conducted to evaluate the system’s ability and effectiveness in delivering precise emotions to audiences. This work has potential applications in artistic drone shows and live concerts.

While prior work [https://arxiv.org/abs/2312.01059] has linked emotions to swarm behaviors or focused on beat-aligned drone performances, few systems model how emotions evolve dynamically within a song, nor do they capture the tension within a song. We aim to replicate this emotional sensitivity in a robotic swarm context.

## json

All our data is kept in the ```/json``` folder

## New Pipeline

Audio → segment the song into verse, chorus, etc. [DONE] then chop it down further → CLAP (maybe put some controllable knobs here too) https://huggingface.co/laion/larger_clap_general→ list(timestamps, descriptions)

Audio → Valence-arousal (need to adjust)→ list(timestampls, emotion weights) [basically what we had before]

Throw both the descriptions and emotion weights into some LLM and have it make a story. (This part would absolutely need some control measures). 

Extract important nouns from the story, then use Text-to-3D models with hopefully vertices (they have some models and datasets on HuggingFace) to generate the drone positions for the visuals

Then we can use the motion principles to control how to shift from one visual to another


## Valence-arousal to Emotion
This part is implemented in
```va.ipynb```

## segmentation
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

## Current challenges

- not quite enough emotional shiftings to make the choreography interesting, especially for sad emotions (it's just spinning in circles)

## Sources

https://drive.google.com/drive/folders/1biQv9RM5Vp0RZNin3hbTd6s_QeQOvaPC?usp=drive_link

## Todo list

Sprint 1:

complete the 
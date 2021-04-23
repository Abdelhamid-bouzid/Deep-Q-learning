# Dueling Network Architectures for Deep Reinforcement Learning.

Dueling Network Architectures for Deep Reinforcement Learning:

"https://arxiv.org/abs/1511.06581"

This folder contains the implementation of dueling Deep Q learning.

As a an application we used cart pole problem from gym envirment.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
What Changes & Motivation
(Wang et al.) presents the novel dueling architecture which explicitly separates the representation of state values and state-dependent action advantages via two separate streams.
The key motivation behind this architecture is that for some games, it is unnecessary to know the value of each action at every timestep. The authors give an example of the Atari game Enduro, 
where it is not necessary to know which action to take until collision is imminent.

Atari Enduro. Source: "https://gfycat.com/clumsypaleimpala"
By explicitly separating two estimators, the dueling architecture can learn which states are (or are not) valuable, without having to learn the effect of each action for each 
state. Like the Enduro example, this architecture becomes especially more relevant in tasks where actions might not always affect the environment in meaningful ways.

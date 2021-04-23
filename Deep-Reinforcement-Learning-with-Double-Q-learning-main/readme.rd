In this folder we implemented the code related to paper entitled: 
   "Deep Reinforcement Learning with Double Q-learning"
   
 The link of the paper is: "https://arxiv.org/abs/1509.06461"
 
 In this paper two deep learning models used to train the Q(s,a) function.
 
 As a problem we used "Atari pong version 4" from gym enviremont.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The contribution od this paper is to address the issue of the overestimation (especially in states we do not wish to learn more about).
The issue is because the DQN uses the same network both to select and to evaluate an action for the next state. To address this issue this paper
proposes a double q learning netwroks where:
  - one is used to evaluate current state and to select action in both:
      - current state
      - next state
  - one is used to evalute next state

  ############# instead of ################
  - one is used to evaluate current state and to select current action
  - one is used to evalute next state and select next action

To summerize the difference is which takes action in the next state 
  

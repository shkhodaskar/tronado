---
layout: default
title: Proposal
---
##Summary of Project
Goal: Stay alive (possibly kill others)
Input: Game State(Pixel value)
Output: A move (direction, up, left, right)

Grid size is 20 by 20
For our project, we will be creating a tron agent for the ColosseumRL competition trained through reinforcement learning. The goal of our project is to create an agent that will survive the longest in the competition. The input for the agent is a list of valid_actions and an observation. In the observation, we are always player 0. The agent picks “the best” valid action based on the observation. Potential applications include self driving cars in terms of avoiding crash, real time bidding etc.

##AI/ML Algorithms

Self-play,Proximal Policy Optimization(PPO) Policy Gradient

For our project, we will be using self play to train the agent and modify its policy using PPO or Policy Gradient.



##Evaluation Plan
Success determined by highest rank
Or best score (score can be determined by how long the agent stays alive or if it kills another agent)
Agent_v1
not run into itself or walls
Agent_v2
surviving longer (stays alive until all/most of the blocks are filled)
Agent_v3
blocking off large areas to kill enemies
Agent_v4
world domination

##Quantitative Evaluation: 
Metrics - time it survives, tries to maximize its number of moves
We expect the agent’s score to be really low at first, maybe 2-3 moves. But as we continue to train it, we expect its score to increase. We will give the agent a reward for every tile it moves. After it is able to survive for 10+ move, we start to add reward for 


##Qualitative Evaluation:
We plan on using self play to teach the agent how to increase its lifespan. We will start by giving the player awards for staying alive longer. But as it learns to not run into itself or into walls, we will start decreasing that reward and increasing the reward for avoid enemy players’ tails. Once our agent gets good enough at avoiding enemies, we can start to increase our reward for blocking off enemy agents.Ultimately we will evaluate by taking part in the UCI RL Tron game competition competing against other agents and hopefully win the competition(moonshot)
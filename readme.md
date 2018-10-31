# Reinforcement learning introduction

A collection of python implementations of the RL algorithms for the examples and figures in Sutton & Barto, Reinforcement Learning: An Introduction.

* Numbering of the exmples is based on the January 1, 2018 complete draft to the 2nd edition.


## Implemented algorithms

### Chapter 2 -- Multi-armed bandits
* Epsilon-greedy action-value methods
* Upper-Confidence-Bound action selection
* Gradient bandit algorithms

### Chapter 3 -- Finite Markov Decision Processes
* State-value function estimation under uniform and optimal policy

### Chapter 4 -- Dynamic programming
* Iterative policy evaluation
* Policy iteration
* Value iteration

### Chapter 5 -- Monte Carlo methods
* First-visit MC
* Exploring starts MC
* Off-policy prediction via importance sampling

### Chapter 6 -- Temporal-Difference learning
* TD(0)
* Batch updating TD(0) and constant-alpha MC
* Sarsa on-policy TD control
* Q-learning off-policy TD control
* Expected Sarsa
* Double Q-learning

### Chapter 7 -- n-step bootstrapping
* n-step TD
* n-step Sarsa

### Chapter 8 -- Planning and learning with tabular methods
* Tabular Dyna-Q
* Planning and non-planning Dyna-Q
* Dyna-Q+ prioritized sweeping for deterministic environments
* Trajectory sampling

### Chapter 9 -- On-policy prediction with approximation
* Gradient Monte Carlo
* Semi-gradient TD(0)
* n-step semi-gradient TD
* Gradient MC with Fourier and polynomial bases
* Coarse coding
* Tile coding
* State aggregation

### Chapter 10 -- On-policy control with approximation
* Episodic semi-gradient Sarsa
* n-step semi-gradient Sarsa
* Differential semi-gradient Sarsa

### Chapter 11 -- Off-policy methods with approximation
* Semi-gradient off-policy TD
* Semi-gradient DP
* TD(0) with gradient correction (TDC)
* Expected TDC
* Expected Emphatic TD

### Chapter 12 -- Eligibility traces
* Offline λ-return
* TD(λ)
* True online TD(λ)
* Sarsa(λ)

### Chapter 13 -- Policy gradient methods
* REINFORCE
* REINFORCE with baseline

A full list of the generated figures and table is [here](figures).


## Usage
Easiest way to run is to clone this repo and run

```
python filename.py
```

## Dependencies
* python 3.6
* numpy
* scipy
* matplotlib
* seaborn
* tqdm
* tabulate

> The key examples of each chapter are separated. There are inter-chapter dependences as examples are extended across topics. Base classes for an base RL agent, Gridworld and tile coding are separated and imported where relevant.



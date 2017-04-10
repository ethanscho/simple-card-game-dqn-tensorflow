# Simple Card Game A3C Tensorflow
This project is an application of DQN/Double DQN to win a simple card game. 

## Purpose
DQN/Double DQN for solving a simple card game. 
In this game, agent will have 3 cards numbered 1, 2, 3 respectively. On the table, there will be 2 random cards among 1, 2, 3. The agent should hand in the same numbered card to get the cards on the table. The agent will have 3 trials. If the agent collects all cards in two turns, the agent will have +1 as a reward, if not the agent will get -1. 

Result graph shows that Double DQN outperforms DQN.
![alt tag](https://github.com/ethanscho/simple-card-game-dqn-tensorflow/blob/master/result.png)

## Usage
1. git clone https://github.com/ethanscho/simple-card-game-a3c-tensorflow.git

2. cd simple-card-game-a3c-tensorflow

3. python a3c.py

## History
v1.0 - Apr. 10. 2017 

## Credits
Ethan Cho.
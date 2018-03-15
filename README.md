고전강화학습에서부터 DQN까지의 강화학습 알고리즘의 이론 및 구현

작성자 : 정원석

언어는 python을 사용하였다. 
deep learning 프레임워크로는 Tensorflow 또는  Kears를 사용하였다. 



---

# 목차

1. 강화학습이란? 
 
 이론: 
 
 
 실습 : 
 
 * openAI tutorial
https://wonseokjung.github.io//reinforcementlearning/update/openai-gym/
 

2. Markov Decision Process
 이론:
 
 실습:
. Dynamic programming

* Dynamic programming Policy Evaluation
* Dynamic programming Policy Iteration
* Dynamic programming Value Iteration

3. MonteCarlo 

 이론:
 * https://wonseokjung.github.io//reinforcementlearning/update/MonteCarlomethod/
 * https://wonseokjung.github.io//reinforcementlearning/update/MC2/
 * https://wonseokjung.github.io//reinforcementlearning/update/RL-MC3/
 * https://wonseokjung.github.io//reinforcementlearning/update/RL-MC4/
 * 
 실습:
 * Get familiar with the Blackjack environment (Blackjack-v0)
 
 * Monte Carlo Prediction to estimate state-action values 
 * on-policy first-visit Monte Carlo Control algorithm 
 * off-policy every-visit Monte Carlo Control using Weighted Important Sampling algorithm 
 

4.  Temporal-Difference Learning
 이론: 
 * one-step TD
 * https://wonseokjung.github.io//reinforcementlearning/update/RL-TD1/
 * https://wonseokjung.github.io//reinforcementlearning/update/RL-TD2/
 * n-step bootstrapping: 
 * https://wonseokjung.github.io//reinforcementlearning/update/RL-NTD1/
 * https://wonseokjung.github.io//reinforcementlearning/update/RL-NTD2/
 * https://wonseokjung.github.io//reinforcementlearning/update/RL-NTD3/
 * Eligibility Traces
 실습:
 * Get familiar with the Windy Gridworld Playground
 * Implement SARSA 
 * Get familiar with the Cliff Environment Playground
 * Implement Q-Learning in Python 

5. Function Approximation
 이론: 
 * On-policy Prediction with Approximation
 
 * On-policy Control with Approximation
 
 실습: 
 * Get familiar with the Mountain Car Playground
 * Q-Learning with Value Function Approximation

6. Deep-Q-Learning
 이론:
 * DQN 
 * DDQN
 * Prioritized Experience Replay 
 실습:
 * Get familiar with the OpenAI Gym Atari Environment Playground
 * Deep-Q Learning for Atari Games 
 * Double-Q Learning 
 * Prioritized Experience Replay 
 * SuperMario-DQN
 * Using Keras and Deep Q-Network to Play FlappyBird
  https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html

7. Policy Gradient Methods
 이론:
 
 실습:
 * REINFORCE with Baseline 
 * Actor-Critic with Baseline 
 * Actor-Critic with Baseline for Continuous Action Spaces 
 * Deterministic Policy Gradients for Continuous Action Spaces (WIP)
 * Deep Deterministic Policy Gradients (WIP)
 * Asynchronous Advantage Actor-Critic (A3C) 








1.네시간 강의 커리큘럼

주제

 Dynamic Programming(Value iteration, Policy iteration ) 

내용

Dynamic programming은 Markov Decision Process(MDP)와 같이 주어진 환경에서 최적의 policy를 계산하기 위해 사용되는 알고리즘 입니다. 

강화학습 문제에서 사용하기에 제약이 많지만 이론적으로 굉장히 중요합니다. 

그리드월드의 예제로 실습을 통하여 Dynamic programming을 사용하여 최적의 policy를 찾아보고, 강화학습 문제에 적용하기에 제약이 많은 이유를 알아보겠습니다. 


- - -


2.

주제

 Monte Carlo method, Temporal-Difference Learning (Sarsa, Q-learning)

내용

Monte Carlo Method와 Temporal-Difference  Learning는 Dynamic programming처럼 

환경의 정보를 알고 시작하는 것이 아닌,  경험을 통해 환경과 상호작용을 하며 배웁니다.  Episode 끝까지 가야지만 Value를 측정 할 수 있는 Monte Carlo Method와 학습을 하며 Value를 업데이트 하며 배우는 Temporal-Difference Learning의 차이점을 실습을 통하여 알아보겠습니다.  

 
- - -

3.

주제

수퍼마리오 환경 구축 및 딥러닝 프레임워크 케라스 소개

내용


딥러닝을 강화학습에 연결하여 이미지와 같은 RGB 데이터를 input으로 받는것이 가능해졌습니다. 파이썬으로 구현된 간결한 딥러닝 라이브러리인 케라스를 사용하여 컨볼루션 신경망 모델을 구현해 보고, 인공지능 슈퍼마리오를 만들기 위해 환경 구축을 하겠습니다.


- - -


4.
주제

DQN을 이용한해 인공지능 수퍼마리오 만들기 

내용

딥러닝을 강화학습에 연결하면서, 여러가지 문제가 발생하였습니다. 
Deepmind는 이 문제를 어떠한 방법으로 해결하여 일부 게임에서 사람보다 플레이를 잘하는 에이전트를 만들었는지 알아보겠습니다.
그리고 그 방법을 수퍼마리오 환경에 적용하여 파이썬과 케라스를 이용한 실습으로 스스로 행동하는 수퍼마리오를 만들겠습니다.
 
- - -



## Reference 
* Reinforcement Learning: An Introduction Richard S. Sutton and Andrew G. Barto Second Edition, in progress
MIT Press, Cambridge, MA, 2017



* Dennnybrtiz
 https://github.com/dennybritz




































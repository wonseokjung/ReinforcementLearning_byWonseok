
목차

1. 강화학습 기초 이론 및 슈퍼마리오 구성 요소

1.1 . Envrionment 

1.2 Emulator 

1.3. Algorithm 

1.4. 프로그래밍 언어 

2. 설치 매뉴얼    

2.1 python 설치

2.2 Emulator 설치

2.3 Environment설치 

3. Algorithm 설명 

4. 오류 해결

5. 슈퍼마리오 훈련

5.1 openAI 함수 설명

5.2 슈퍼마리오의 action

5.3 코드 설명

6. 대회규칙





_ _ _

## 1. 강화학습 기초 이론 및 슈퍼마리오 구성 요소

똑똑한(?) 슈퍼마리오를 만들기 위해서는 4가지가 필요합니다.

### 1.1 . Envrionment 
![image1](https://user-images.githubusercontent.com/11300712/37243505-b82841f8-24bd-11e8-99a0-a69fe6faa822.jpg)


Agent인 마리오가 Environment와 상호작용을 하며 환경에서 주어진 action중 하나를 선택하고 그 action을 선택함으로써 reward(보상) 을 받게 됩니다.

이러한 Frame을 MDP라고 합니다.

MDP는 목표를 달성하기 위해 상호작용하는 큰 frame입니다.

여기서 배우고 결정을 내리는 것을 Agent 라고 합니다.

Agent와 상호작용하는 것, agent를 제외한 모든 것을 Environment라고 합니다.

Agent는 각 time step $$t$$마다 환경을 표현하는 state $$s$$를 받습니다. 
 
State를 받고 Environment가 상호작용을 하면서 agent가 action을 선택하면,

Environment는 agent의 action에 응답해 그에 맞는 새로운 상황과 reward를 agent 에게 줍니다.

강화학습을 이용하여 스스로 높은 reward를 받는 action을 선택하는 슈퍼마리오를 만드는 것이 목표입니다. 

이러한 정보가 있는 “슈퍼마리오”의 Environment가 필요합니다. 


### 1.2 Emulator 

두번째로 슈퍼마리오를 실행하기 위한 Emulator가 필요합니다. 

강화학습에서는 Agent가 현재의 state에서 action을 선택하고, 

그 action을 하여 받는, 다음의 state와 reward 정보를 받으며 학습을 합니다. 


여기서 Agent는 마리오이고 state는 게임 화면입니다. 

마리오는 이미지 데이터인 게임 화면을 matrix로 바꿔 state를 인식하고 이것을 이용하여 학습을 합니다. 

그렇게 하기 위해서는 슈퍼마리오 게임을 구동하는 프로그램이 필요한데요. 

이 메뉴얼에서 우리는 fceux라는 emulator을 사용할 것입니다.


### 1.3. Algorithm 

강화학습에서 목표는, Agent가 학습을 하며 받는 총 reward의 크기를 최대화하는 것입니다. 
바로 앞에서 받을수 있는 reward를 최대화 시키는 것이 아닌 long run에서의 축척된 reward를 최대화시키는 것 입니다.
그럼 reward를 최대화 시키기 위해 그 환경에 맞는 적절한 학습 알고리즘을  사용하여야 합니다. 

Tensorflow 라이브러리를 사용하였습니다. 

딥러닝 모델은 vgg를 사용하였고, 

### 1.4. 프로그래밍 언어 

환경을 구성하고 강화학습의 알고리즘을 구현하기 위해 Python 프로그래밍 언어를 사용합니다. 

그리고 

이 메뉴얼은 Window 환경에서는 슈퍼마리오 환경, emulator와 충돌이 많아 Ubuntu와 Mac을 기준으로 작성하였습니다. 

UBUNTU, MAC에서 실습하실것을 강력 추천 드립니다( 윈도우에서 성공한 사례는 보지 못하였습니다.)


## 2. 설치 매뉴얼

### 2.1 python 설치

![image11](https://user-images.githubusercontent.com/11300712/37243743-ea42d1ae-24c1-11e8-9310-2ccad3af6ecd.jpg)

a.첫번째로 파이썬을 설치하셔야 합니다. 

https://www.python.org/ - python 홈페이지 

위의 홈페이지에 들어가셔서 Mac 혹은 우분투 버전을 다운로드 후 설치해주세요. 

파이썬은 버전이 여러가지 있는데, 

현재 파이썬 3.5버전에서의 실행을 확인하였습니다. 

그러므로 파이썬 3.5버전 설치를 권장합니다. 


b.그리고, 딥러닝 모델을 쉽게 구현할 수 있는 라이브러리인 케라스와 텐서플로우를 설치해주세요. 
https://www.tensorflow.org/install/ - Tensorflow 

Ubuntu ctrl+alt+t 를 눌러 커맨창에서 

tensorflow 일반 버전은

`sudo pip3 install tensorflow `

혹시 gpu가 있으시다면, 

`sudo pip3 install tensorflow-gpu`  명령어로 tensorflow를 설치해주세요. 

케라스 홈페이지를 다음과 같으며 케라스 설치하는 방법이 자세히 설명되어 있습니다. 

https://keras.io/#installation

c. Jupyter notebook을 설치해주세요. 

실습편에서 jupyter notebook을 이용하여 코드 설명을 하였습니다. 

이 파일을 실행시키기 위하여 jupyter notebook이 필요합니다. 

http://jupyter.org/install

`python3 -m pip install --upgrade pip`
`python3 -m pip install jupyter`

위의 명령어를 이용하여 설치하여 주세요. 


## 2.2 Emulator 설치

![image5](https://user-images.githubusercontent.com/11300712/37247381-3728c70e-24fd-11e8-9d5e-336b0d697e1f.jpg)



Emulator인 fceux는 슈퍼마리오 게임을 실행시켜주는 역할을 합니다. 

fceux홈페이지에 들어가시면 emulator에 관한 자세한 설명이 나와있습니다. 
http://www.fceux.com/web/home.html 

ubuntu 혹은 Mac에서 커맨드 창에 다음의 명령어를 입력하여 fceux를 설치하여주세요. 

Ubuntu 

`sudo apt-get update`
`sudo apt-get install fceux`

MAC 

https://brew.sh/  - homebrew 웹사이트

커맨드 창에 아래의 명령어 입력
`brew install fceux`
`sudo apt-get install fceux`

## 2.3 Environment설치 

![image3](https://user-images.githubusercontent.com/11300712/37247379-3069fc1c-24fd-11e8-9f2c-4196db6bf4a5.jpg)

Openai 에서 gym 과 baselines를 제공해 줍니다. 

우리는 이를 이용하여 강화학습의 환경을 다운 받고 학습 알고리즘을 실험해 볼 수 있습니다. 

openai에서 슈퍼마리오의 환경은 제공하고 있지 않지만, 

우리는 슈퍼마리오의 환경을 gym의 폴더에 넣어 openai가 제공하는 함수를 사용하기 위해 먼저 openAI의 gym 라이브러리를 다운받아야 합니다. 



https://openai.com/   - Openai 

https://github.com/openai/gym - gym git 주소





a.필요한 package를 설치하세요.

MAC

`brew install cmake boost boost-python sdl2 swig wget`

Ubuntu

`apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig`


b.gym 설치


ubuntu

pip를 이용해서 설치하기

`pip3 install gym` 

혹은 git에서 다운받아 설치하기 

`git clone https://github.com/openai/gym.git`
`cd gym`
`pip3  install -e`


baselines에 DQN,A3C등 여러가지 알고리즘의 예제가 나와있습니다. 

우리는 baselines를 직접 사용하진 않지만 강화학습 알고리즘 분석을 위해 다운받길 권장합니다. 


Baselines

https://github.com/openai/baselines

pip를 이용해서 설치하기

pip3 install baselines


혹은 git에서 다운받아 설치하기 

git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .


![image5](https://user-images.githubusercontent.com/11300712/37247381-3728c70e-24fd-11e8-9d5e-336b0d697e1f.jpg)


다음은 슈퍼마리오의 환경을 다운받아야 합니다. 

강화학습으로 슈퍼마리오를 학습시킬 수 있도록, 

Philip Paquette가 슈퍼마리오와 둠의 환경을 github에 업로드 해놓았습니다. 

우리는 gym-pull이란 함수를 이용하여 philip paquette가 만든 슈퍼마리오 환경을 위에서 설치한 gym의 환경에 넣어줍니다. 


philip paquette의 github 주소

https://github.com/ppaquette/gym-super-mario

커맨드창을 여신후에 

a.pip를 이용하여 gym-pull을 설치하세요 

`pip3 install gym-pull`

b. 커맨드창에서 python3을 이용하여 python3에 들어가신 후 gym과 gym_pull을 import 하셔야합니다.  
- sudo python3 명령어를 이용하여 python3에 들어가세요 
- python3에서 gym과 gym_pull 라이브러리를 import 합니다. 
`import gym`
`import gym_pull`

c. gym_pull을 이용하여 ppaqutte의 supermario 환경을 가져옵니다. 

`gym_pull.pull('github.com/ppaquette/gym-super-mario')`       

d.gym.make의 함수를 이용하여 슈퍼마리오 환경이 로드 되는지 확인합니다. 

`env = gym.make('ppaquette/SuperMarioBros-1-1-v0')`

* gym_pull error가 발생하시는 분은 메뉴얼 목차 4의 오류발생 부분을 확인해주세요.
 
## 3.Algorithm 설명 

![image9](https://user-images.githubusercontent.com/11300712/37243738-db1a797a-24c1-11e8-8be6-d68387531fca.jpg)

저는 학습알고리즘으로 DQN을 사용하였습니다. 
DQN은 replay memory를 이용하여 state,action,reward, next state의 정보를 memory에 저장하고, 이 정보를 이용하여 convolutional neural network라는 딥러닝 모델을 사용하여 좋은 action을 선택하게하는 알고리즘 입니다. 

더 자세한 설명은 아래의 링크를 참조하세요. 

[DQN이란 무엇일까요? ](https://wonseokjung.github.io//rl_paper/update/RL-PP-DQN/)


참고로 저의 경우, 레벨1을 클리어하기 위하여 5000 에피소드 정도의 학습이 필요하였습니다.

DQN을 사용한 것은 단 하나의 예 입니다. 

환경에 맞는 적절한 강화학습 알고리즘을 사용하는 것이 중요하므로, 학습 알고리즘은 본인의 결정에 따라 자유롭게 선택하시면 됩니다. 


## 4. 오류 해결


A. 
설치하며 여러가지 오류가 발생하였는데요. 그중 가장 많이 발생한 gym_pull 오류를 풀잎1기 강화학습반 김경환님이 해결하신 방법입니다.

![image10](https://user-images.githubusercontent.com/11300712/37247567-60d718d2-2500-11e8-878c-451805bff4f9.jpg)
![image13](https://user-images.githubusercontent.com/11300712/37247569-6c597786-2500-11e8-801f-c690fa884118.jpg)
![screenshot from 2018-03-11 08-55-45](https://user-images.githubusercontent.com/11300712/37248031-036d15fc-250a-11e8-9827-df778a9d74e7.png)
![image12](https://user-images.githubusercontent.com/11300712/37248033-0bc40b34-250a-11e8-9fe4-7d04d0f3ae22.jpg)

![screenshot from 2018-03-11 08-56-36](https://user-images.githubusercontent.com/11300712/37248040-4afe19ac-250a-11e8-93d0-1a7060586067.png)


B. 풀잎2기 강화학습반 권력환님의 gym_pull 을 사용하지 않고 환경 가져오는 방법


gym & super mario 환경 설치

gym-0.9.4 설치
(gym이 이미 깔려있는 경우..)
`pip3 uninstall gym` 
`pip3 install gym==0.9.4 ` 

baseline 설치
`pip3 install baselines`

Super Mario Gym 환경 설치 
`git clone https://github.com/ppaquette/gym-super-mario.git`
`cd gym-super-mario/`
`pip3 install -e .`

Super Mario Gym 환경 불러오기

`python3 -V`
Python 3.5.2

`python3`
`import gym`
`import ppaquette_gym_super_mario`
`env = gym.make('ppaquette/SuperMarioBros-1-1-v0')`
`env.reset() `
`env.close()`


## 5. 슈퍼마리오 훈련

### 5.1 슈퍼마리오에서의 openAI 함수 사용법

* `env=gym.make('ppaquette/meta-SuperMarioBros-v0')`

gym의 make함수를 사용하여 슈퍼마리오 환경을 불러옵니다. 


* `env.reset()`

env.reset()을 통하여 emulator인 fceux가 실행됩니다.

마리오가 죽으면 게임이 종료되고 게임을 시작하면 일정시간 내에 목표지점인 깃발까지 도착해야 합니다.

총 reward는 목표까지의 X측의 거리 입니다. (깃발까지 가깝게 가면 갈수록 더 많은 reward를 받습니다.)


* `env.step()`

env.step() 함수를 통하여 마리오가 action을 선택하면

`env.step(action)`

obs, reward, is_finished, info 의 값을 받습니다.

obs : 그 action을 선택하여 받은 관찰값 입니다.

reward: 그 action을 선택하여 받은 보상값

is_finished : 마리오가 죽었는지 아닌지에 대한 정보입니다.

* info

distance : 마리오가 X축으로 이동한 거리

life : 마리오가 현재 가지고 있는 생명의 횟수

score : 현재 스코어

coins : 현재 가지고 있는 코인의 갯수

time : 현재 남은 시간의 정보




player_status : 작은마리오일때 -> 0 , 큰 마리오일때 ->1, 불꽃을 쏘는 마리오일때 ->2 의 값을 가지고있습니다.

* 	`env.action_space.sample()`

임의의 action을 선택합니다. 

* `env.observation_space` 

현재 state의 이미지 사이즈를 출력합니다. 


### 5.2 슈퍼마리오의 action

마리오는 다음과 같은 배열로 action을 인식합니다.
Up, Left, Down, Right, A, B

아무 action도 선택하지 않는 상태 : [0,0,0,0,0,0]

위로가는(방향키 위)버튼을 선택한다 : [1,0,0,0,0,0]

왼쪽으로 가는 action을 선택한다. : [0,1,0,0,0,0]

아래로 가는 action을 선택한다. : [0,0,1,0,0,0]

오른쪽으로 가는 action을 선택한다 : [0,0,0,1,0,0]

점프를 하는 action을 선택한다(A버튼) : [0,0,0,0,1,0]

버튼을 눌러 불꽃을 쏘는 action을 선택한다(B버튼): [0,0,0,0,0,1]

action = [0, 0, 0, 1, 1, 0] # [up, left, down, right, A, B]

이렇게 두개 버튼을 같이 선택한다면 4번째( 오른쪽 ) 와 5번째( 점프 )의 action을 같이 할것입니다.

이와같이 6가지 action을 선택하며 환경을 통해 reward를 받습니다.



### 5.3 코드 설명

* 환경 불러오기
* 
```
import gym
import ppaquette_gym_super_mario
env = gym.make('ppaquette/meta-SuperMarioBros-v0')
```
슈퍼마리오 환경을 임포트 합니다.
gym의 함수인 make 함수를 사용하여 위에서 불러온 환경을 현재 환경으로 지정합니다.

* 여러가지 레벨

ppaquette의 슈퍼마리오는

총 32개 레벨의 마리오 환경을 제공하며 우리는 'ppaquette/meta-SuperMarioBros-v0'환경으로
학습을 하도록 하겠습니다.

meta-SuperMarioBros-v0 : 

레벨 클리어시 자동으로 다음 레벨로 넘어갑니다. 

ppaquette/SuperMarioBros-1-1-v0:

레벨1 클리어하더라도 다시 레벨1로 리셋됩니다. 

ppaquette/SuperMarioBros-1-1-Tiles-v0:

Tiles이 붙은 환경은 이미지의 사이즈가 13 x 16 x 1로 일반 환경에 비해 매우 작습니다. 

아래의 64개의 환경중 원하는 환경을 쓰시면 됩니다!

ppaquette/meta-SuperMarioBros-v0
ppaquette/meta-SuperMarioBros-Tiles-v0
ppaquette/SuperMarioBros-1-1-v0
ppaquette/SuperMarioBros-1-2-v0
ppaquette/SuperMarioBros-1-3-v0
ppaquette/SuperMarioBros-1-4-v0
ppaquette/SuperMarioBros-2-1-v0
ppaquette/SuperMarioBros-2-2-v0
ppaquette/SuperMarioBros-2-3-v0
ppaquette/SuperMarioBros-2-4-v0
ppaquette/SuperMarioBros-3-1-v0
ppaquette/SuperMarioBros-3-2-v0
ppaquette/SuperMarioBros-3-3-v0
ppaquette/SuperMarioBros-3-4-v0
ppaquette/SuperMarioBros-4-1-v0
ppaquette/SuperMarioBros-4-2-v0
ppaquette/SuperMarioBros-4-3-v0
ppaquette/SuperMarioBros-4-4-v0
ppaquette/SuperMarioBros-5-1-v0
ppaquette/SuperMarioBros-5-2-v0
ppaquette/SuperMarioBros-5-3-v0
ppaquette/SuperMarioBros-5-4-v0
ppaquette/SuperMarioBros-6-1-v0
ppaquette/SuperMarioBros-6-2-v0
ppaquette/SuperMarioBros-6-3-v0
ppaquette/SuperMarioBros-6-4-v0
ppaquette/SuperMarioBros-7-1-v0
ppaquette/SuperMarioBros-7-2-v0
ppaquette/SuperMarioBros-7-3-v0
ppaquette/SuperMarioBros-7-4-v0
ppaquette/SuperMarioBros-8-1-v0
ppaquette/SuperMarioBros-8-2-v0
ppaquette/SuperMarioBros-8-3-v0
ppaquette/SuperMarioBros-8-4-v0
ppaquette/SuperMarioBros-1-1-Tiles-v0
ppaquette/SuperMarioBros-1-2-Tiles-v0
ppaquette/SuperMarioBros-1-3-Tiles-v0
ppaquette/SuperMarioBros-1-4-Tiles-v0
ppaquette/SuperMarioBros-2-1-Tiles-v0
ppaquette/SuperMarioBros-2-2-Tiles-v0
ppaquette/SuperMarioBros-2-3-Tiles-v0
ppaquette/SuperMarioBros-2-4-Tiles-v0
ppaquette/SuperMarioBros-3-1-Tiles-v0
ppaquette/SuperMarioBros-3-2-Tiles-v0
ppaquette/SuperMarioBros-3-3-Tiles-v0
ppaquette/SuperMarioBros-3-4-Tiles-v0
ppaquette/SuperMarioBros-4-1-Tiles-v0
ppaquette/SuperMarioBros-4-2-Tiles-v0
ppaquette/SuperMarioBros-4-3-Tiles-v0
ppaquette/SuperMarioBros-4-4-Tiles-v0
ppaquette/SuperMarioBros-5-1-Tiles-v0
ppaquette/SuperMarioBros-5-2-Tiles-v0
ppaquette/SuperMarioBros-5-3-Tiles-v0
ppaquette/SuperMarioBros-5-4-Tiles-v0
ppaquette/SuperMarioBros-6-1-Tiles-v0
ppaquette/SuperMarioBros-6-2-Tiles-v0
ppaquette/SuperMarioBros-6-3-Tiles-v0
ppaquette/SuperMarioBros-6-4-Tiles-v0
ppaquette/SuperMarioBros-7-1-Tiles-v0
ppaquette/SuperMarioBros-7-2-Tiles-v0
ppaquette/SuperMarioBros-7-3-Tiles-v0
ppaquette/SuperMarioBros-7-4-Tiles-v0
ppaquette/SuperMarioBros-8-1-Tiles-v0
ppaquette/SuperMarioBros-8-2-Tiles-v0
ppaquette/SuperMarioBros-8-3-Tiles-v0
ppaquette/SuperMarioBros-8-4-Tiles-v0



* 4장의 화면을 붙여 마리오게 적과 함정 

슈퍼마리오는 화면을 보고 적의 위치와 함정등의 정보를 받아옵니다.

만약 슈퍼마리오에게 화면이 멈춰진 한장면만 주어진다면, 적이 오는 것을 판단하기가 힘들것 입니다.

그래서 멈춰진 화면 4개를 연속적으로 주어 마리오에게 적의 위치와 움직이는 방향을 판단할수 있게 해줘야합니다!

```
obs=env.reset()
print(obs)
reshape_obs=np.reshape([obs],(1,84,84,1))
history=np.stack((obs,obs,obs,obs), axis = 2)
history = np.reshape([history], (1, 84, 84, 4))
history=np.append(reshape_obs, history[:,:,:,:3], axis=3)
processed_obs=np.reshape([history],(84,84,4))
```

* 마리오가 E-epsilon 행동을 하게 만들기
* 
greedy 한 action 을 선택하게 해주는 함수를 만들어 줍니다.

numpy의 np.random.rand()함수를 통해 epsilon 값보다 적은 값이 나올땐 0 부터 5까지의 임의의 값을 리턴해주고,

epsilone 보다 큰 값이 나왔을때는 keras.predict q value 의 값을 가장 크게 만드는 action을 리턴하게 합니다.
```
epsilon=0.9
action_size=6

import random

def get_action(history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= epsilon:
            return random.randrange(action_size)
        else:
            q_value = model.predict(history)
            return np.argmax(q_value[0])

action=get_action(history)
            
```
            

* 임의의 임의의 action을 선택하는 mario 만들기

임의의 action을 선택하는 

## 6. 대회규칙

슈퍼마리오는 총 0레벨부터 31 레벨까지 있고,  

대회의 목표는 각 레벨의 끝에 있는 깃발을 잡는것이 목표입니다. 

게임의 score( 코인, 적을 제거 등 )은 고려하지 않으며, x축으로의 이동거리로 승부를 결정합니다.

만약 레벨1을 클리어하는 마리오가 두개 이상 발생한다면, 이어서 다음레벨에서의 X축의 이동거리로 승부를 결정합니다.

레벨 클리어시 다음 레벨로 자동으로 넘어갈 수 있게, 
환경은 ppaquette/meta-SuperMarioBros-v0를 사용해주세요. 

학습 알고리즘과 딥러닝 모델의 선택은 개인자유입니다. 





### 예제코드 

* Keras 
[DQN, Keras model](https://github.com/wonseokjung/moduyeon_supermario/tree/master/keras_model)

Keras를 이용하여 슈퍼마리오를 학습하였습니다.

Jupyter notebook을 이용한, 코드 설명이 포함되어 있습니다. 
* Tensorflow
[DQN, Tensorflow, Episode 5900](https://github.com/wonseokjung/moduyeon_supermario/tree/master/tensorflow)

위의 github링크에 DQN을 사용하여 슈퍼마리오를 훈련시킨 코드를 올려놓았습니다. 
저장되어 있는 모델을 리스토어하시면, 슈퍼마리오가 5900정도의 에피소드를 훈련하였을때 어떤 action을 선택하는지 체험하실수 있습니다. 
장종성님의 도움을 받았습니다. 




### References

https://github.com/wonseokjung/moduyeon_supermario/tree/master/tensorflow


https://wonseokjung.github.io//rl_paper/update/RL-PP-DQN/


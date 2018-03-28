




## Atar 설치법

$ pip install Pillow
$ pip install numpy
$ pip install tensorflow==1.0.0
$ pip install keras==2.0.3
$ pip install matplotlib
$ pip install h5py
$ pip install gym
$ pip install gym[atari]
$ pip install -U scikit-learn
 텐서플로,케라스버전은 위와 동일하지 않아도 무관

# osx

sudo pip3 install -U scikit-learn

windows에서 atari를설치하는방법은두가지가있습니다.mysys 를통해서설치하기:http://ishuca.tistory.com/entry/Windows%EC%97%90%EC%84%9C-gymatari-%EC%84%A4%EC%B9%98%ED%95%98%EA%B8%B0bash 를통해서설치하기:https://brunch.co.kr/@gnugeun/28-두방법중하나를선택하셔서설치를진행해주시면감사하겠습니다.


# Break out 환경으로 DQN 강화학습 알고리즘 적용하기 ( 케라스 버전 )







## 1. 전체적인 순서 (Main part)

1. 환경을 불러온다. 

2. agent를 생성한다. 

3. score, episode, global_step 를 정해준다. 

4. 정해준 에피소드만큼 학습을 시작 

- [ ] 목숨 다섯개가 주어진다. 

- [ ] 환경의 초기값을 가져온다. 

- [ ] 처음 일정구간동안 바가 action을 임의로 선택할수 있게 한다. 

- [ ] 위의 환경 초기값을 이미지 전처리 해준다.

- [ ] 전처리해준 이미지로 네개의 히스토리를 만든다. 

* 죽을때까지 계속 돌게하는 while문을 만든다. 

- [ ] render의 유무를 확인한다. (render을 유무에 따라 학습 속도가 달라진다.)

- [ ]글로벌 스탭을 하나씩 늘려준다.

- [ ] 스탭을 하나씩 늘려준다. 

- [ ] 네개의 state( history )를 이용하여 action을 선택한다. 

- [ ] 선택한 action으로 환경과 상호작용하며 환경에서 다음 스테이트, reward, done, info의 값을 받는다.

- [ ] 받은 다음 스테이트를 다시 전처리 해준다. 

- [ ] history 에서 앞의 세개와 방금 받아온  state를 합쳐서 next_history로 선언

- [ ] q_max의 평균을 계산하기 위해서 현재의 model 로 부터 나온 Q 값의 max를 agent.avg_q_max에 더한다.

- [ ] 만약 dead인 경우 dead를 True로 바꾸고, start_life를 하나 줄여준다.

- [ ] 다른 아타리 게임에서도 적용할 수 있도록 리워드의 범위를 -1~1로 한다.

- [ ] s,a,r,s'를 리플레이 메모리에 저장한다. 

- [ ] 리플레이 메모리가 시작점보다 높아지면 학습을 시작한다. 

- [ ] 일정시간마다 target model을 update한다. 

- [ ] 만약에 죽으면 dead = false로 바꾸고 아니면 next history 값을 history가 받는다. 

- [ ] 만약에 done이면 에피소드의 학습정보를 기록하여 출력한다. 

- [ ] 일정 에피소드마다 모델을 저장한다.


## 2.필수 라이브러리 및 함수

1.필요한 라이브러리를 불러온다. 

a.Keras 

* CNN layer

* Dense layer
 
* optimizer
 
* 케라스에서의 딥러닝 모델

b. 이미지 전처리

* input으로 들어오는 이미지 크기 조절

* RGB를 Gray로 만드는 라이브러리

* replay memory만드는

c.Tensorflow  

* tensorflow backend
 
* tensorflow

d. 기타 

* numpy
 
* random
 
* gym
 
* os


- - -

2.Save 저장경로 

3.총 Episode 선언

4.DQN Agent class 생성

a.초기화

* render 의 유무

* model의 load 유무

* state 사이즈

* action 사이즈

* epsilon값

* epsilon의 시작점과 끝점 ( decay를 위해 )

* epsilon의 decay step 정의

* 리플레이 메모리에서 뽑을 배치사이즈 정의

* 학습을 시작할 기준 정의

* 정답 모델로 업데이트 주기 정의

* discount factor

* 리플레이메모리 최대크기 정의

* 스타트할때 action을 임의로 정해주는 설정

* Deeplearning model

* Target model 

* update target model

* optimizer

* Tensorboard 

* load model


b.Optimizer 함수 설정

c. 딥러닝 모델 함수

d. 현재 model의 weight를 가져와서 target model의 웨이트로 업데이트 하는 함수

e. action을 선택하는 함수(policy)

f. state, action, reward, next state를 리플레이 메모리에 저장해주는 함수

g. 리플레이 메모리에서 뽑아온 배치로 모델을 학습하는 함수

h. 각 에피소드 당 학습 정보를 기록하는 함수

5.이미지 전처리를 위한 함수







* * *





## 코드 설명


1.필요한 라이브러리를 불러온다. 

a.Keras 

* CNN layer

* Dense layer
 
* optimizer
 
* 케라스에서의 딥러닝 모델

b. 이미지 전처리

* input으로 들어오는 이미지 크기 조절

* RGB를 Gray로 만드는 라이브러리

* replay memory만드는

c.Tensorflow  

* tensorflow backend
 
* tensorflow

d. 기타 

* numpy
 
* random
 
* gym
 
* os


```
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten
from keras.optimizers import RMSprop
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from keras import backend as K
import tensorflow as tf
import numpy as np
import random
import gym
import os
```








2.Save 저장경로 
```
model_path = os.path.join(os.getcwd(),'save_model')

if not os.path.isdir(model_path):
    os.mkdir(model_path)
```

3.총 Episode 선언

```
EPISODES = 50000
```



4.DQN Agent class 생성

a.초기화

* render 의 유무

* model의 load 유무

* state 사이즈

* action 사이즈

* epsilon값

* epsilon의 시작점과 끝점 ( decay를 위해 )

* epsilon의 decay step 정의

* 리플레이 메모리에서 뽑을 배치사이즈 정의

* 학습을 시작할 기준 정의

* 정답 모델로 업데이트 주기 정의

* discount factor

* 리플레이메모리 최대크기 정의

* 스타트할때 action을 임의로 정해주는 설정

* Deeplearning model

* Target model 

* update target model

* optimizer

* Tensorboard 

* load model





``` 
# 브레이크아웃에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
 
        self.optimizer = self.optimizer()
 
        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
 
        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
 
        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn_trained.h5")
 
    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')
 
        prediction = self.model.output
 
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)
 
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
 
        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)
 
        return train
 
    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2),
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1),
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model
 
    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
 
    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])
 
    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))
 
    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
 
        mini_batch = random.sample(self.memory, self.batch_size)
 
        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []
 
        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])
 
        target_value = self.target_model.predict(next_history)
 
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])
 
        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]
 
    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
 
        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
 
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
```

5.이미지 전처리를 위한 함수

``` 
# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe
```


## 1. 전체적인 순서 (Main part)

1. 환경을 불러온다. 

2. agent를 생성한다. 

3. score, episode, global_step 를 정해준다. 

4. 정해준 에피소드만큼 학습을 시작 

- [ ] 목숨 다섯개가 주어진다. 

- [ ] 환경의 초기값을 가져온다. 

- [ ] 처음 일정구간동안 바가 action을 임의로 선택할수 있게 한다. 

- [ ] 위의 환경 초기값을 이미지 전처리 해준다.

- [ ] 전처리해준 이미지로 네개의 히스토리를 만든다. 

*  죽을때까지 계속 돌게하는 while문을 만든다. 

- [ ] render의 유무를 확인한다. (render을 유무에 따라 학습 속도가 달라진다.)

- [ ]글로벌 스탭을 하나씩 늘려준다.

- [ ] 스탭을 하나씩 늘려준다. 

- [ ] 네개의 state( history )를 이용하여 action을 선택한다. 

- [ ] 선택한 action으로 환경과 상호작용하며 환경에서 다음 스테이트, reward, done, info의 값을 받는다.

- [ ] 받은 다음 스테이트를 다시 전처리 해준다. 

- [ ] history 에서 앞의 세개와 방금 받아온  state를 합쳐서 next_history로 선언

- [ ] q_max의 평균을 계산하기 위해서 현재의 model 로 부터 나온 Q 값의 max를 agent.avg_q_max에 더한다.

- [ ] 만약 dead인 경우 dead를 True로 바꾸고, start_life를 하나 줄여준다.

- [ ] 다른 아타리 게임에서도 적용할 수 있도록 리워드의 범위를 -1~1로 한다.

- [ ] s,a,r,s'를 리플레이 메모리에 저장한다. 

- [ ] 리플레이 메모리가 시작점보다 높아지면 학습을 시작한다. 

- [ ] 일정시간마다 target model을 update한다. 

- [ ] 만약에 죽으면 dead = false로 바꾸고 아니면 next history 값을 history가 받는다. 

- [ ] 만약에 done이면 에피소드의 학습정보를 기록하여 출력한다. 

- [ ] 일정 에피소드마다 모델을 저장한다.


````
if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)
 
    scores, episodes, global_step = [], [], 0
 
    for e in range(EPISODES):
        done = False
        dead = False
 
        step, score, start_life = 0, 0, 5
        observe = env.reset()
 
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)
 
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
 
        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1
 
            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3
 
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)
            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)
 
            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])
 
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
 
            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)
 
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
 
            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()
 
            score += reward
 
            if dead:
                dead = False
            else:
                history = next_history
 
            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
 
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)
 
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))
 
                agent.avg_q_max, agent.avg_loss = 0, 0
 
        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_dqn.h5")
```

### Reference

* https://github.com/wonseokjung/reinforcement-learning-kr

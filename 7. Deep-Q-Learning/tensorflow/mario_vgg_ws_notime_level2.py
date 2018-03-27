


import ppaquette_gym_super_mario
import numpy as np
import tensorflow as tf
import random
from collections import deque
from dqn_mario import dqn

from random import randint




import gym
from gym import wrappers
#reward : distance

checkpoint_dir = './wonseok_checkpoint_level2/'  #
train = True	
retrain = True

env = gym.make('ppaquette/SuperMarioBros-1-2-v0')
env = wrappers.Monitor(env, 'gym-results', force=True)

# Constants defining our neural network

input_size = np.array([env.observation_space.shape[0], env.observation_space.shape[1], 15]) #width*height*3ch
output_size = 13 #up, down, left, right, run, jump

dis = 0.9
REPLAY_MEMORY = 20000

# Minibatch works better

def ddqn_replay_train(mainDQN, targetDQN, train_batch, l_rate):
    '''
    Double DQN implementation
    :param mainDQN: main DQN
    :param targetDQN: target DQN
    :param train_batch: minibatch for train
    :return: loss
    '''
    #x_stack = np.empty(0).reshape(0, mainDQN.input_size)
    x_stack = np.empty(0).reshape(0, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])
    y_stack = np.empty(0).reshape(0, mainDQN.output_size)
    action_stack = np.empty(0).reshape(0, 60)

    # Get stored information from the buffer
    for state, action_seq, action_next_seq, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state, action_seq)

        # terminal?

        if done:
            Q[0, action] = reward
        else:

            Q[0, action] = reward + dis * targetDQN.predict(next_state, action_next_seq)[0, np.argmax(mainDQN.predict(next_state, action_next_seq))]


        if state is None:
            print("None State, ", action, " , ", reward, " , ", next_state, " , ", done)
        else:
            y_stack = np.vstack([y_stack, Q])
            x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size[0]*mainDQN.input_size[1]*mainDQN.input_size[2])])
            action_stack = np.vstack([action_stack, np.reshape(action_seq, (-1, 60))])
            #x_stack = np.vstack([x_stack, state.reshape(-1, mainDQN.input_size)])

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(x_stack, y_stack, action_stack, l_rate = l_rate)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):

    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder

def bot_play(mainDQN, env=env):
    # See our trained network in action
    state = env.reset()
    reward_sum = 0
    while True:

        if state is None or state.size == 1:
            output = randint(0, output_size - 1)
            action = OutputToAction3(output)
            print("random action:", output)
        else:
            output = np.argmax(mainDQN.predict(state))
            action = OutputToAction3(output)
            print("predicted action:", output)
        for n in range(len(action)):
            state, reward, done, info = env.step(action[n])
            if done == True:
                break
        reward_sum += reward
        if done:
            print("Total score: {}".format(reward_sum))
            break


def OutputToAction3(output): #A:jump B:run

    if output == 0:
        action = np.array([[0, 0, 0, 0, 0, 0]]*2)  # NOOP
    elif output == 1:
        action = np.array([[1, 0, 0, 0, 0, 0]]*2)  # Up
    elif output == 2:
        action = np.array([[0, 0, 1, 0, 0, 0]]*2)  # Down
    elif output == 3:
        action = np.array([[0, 1, 0, 0, 0, 0]]*2)  # Left
    elif output == 4:
        action = np.array([[0, 1, 0, 0, 1, 0]]*2)  # Left + A (short jump)
    elif output == 5:
        action = np.array([[0, 1, 0, 0, 0, 1]]*2)  # Left + B
    elif output == 6:
        action = np.array([[0, 1, 0, 0, 1, 1]]*2)  # Left + A + B (short jump)
    elif output == 7:
        action = np.array([[0, 0, 0, 1, 0, 0]]*2)  # Right
    elif output == 8:
        action = np.array([[0, 0, 0, 1, 1, 0]]*2)  # Right + A (short jump)
    elif output == 9:
        action = np.array([[0, 0, 0, 1, 0, 1]]*2)  # Right + B
    elif output == 10:
        action = np.array([[0, 0, 0, 1, 1, 1]]*2)  # Right + A + B (short jump)
    elif output == 11:
        action = np.array([[0, 0, 0, 0, 1, 0]]*2)  # A (short jump)
    else:
        action = np.array([[0, 0, 0, 0, 1, 1]]*2) # A + B (short jump)

    return action

def main():
    if train == True:
        init_episode = 1383
        max_episodes = 10000
        # store the previous observations in replay memory
        replay_buffer = deque()
        state_buffer = deque()
        next_state_buffer = deque()
        output_buffer = deque()


        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        with tf.Session(config=config) as sess:
            mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
            targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
            tf.global_variables_initializer().run()
            copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
            sess.run(copy_ops)

            if retrain == True:
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_dir + 'model_notime_level2.ckpt')
                print("restored!")

            # initial copy q_net -> target_net


            for episode in range(init_episode, max_episodes):
                #e = 1. / ((episode // 100)/10 + 1)
                e = 1.0 / (episode/500 + 1)
                print("episode:", episode, ", e:", e)
                done = False
                step_count = 0
                state = env.reset()
                score = 0
                distance = 0
                prev_output = -1
                repeat =0

                while not done:
                    #if np.random.rand(1) < e:
                    if np.random.rand(1) < e or state is None or state.size == 1 or step_count<=10:
                    #if (np.random.rand(1) < e and episode%2==0) or step_count <= 10:

                        output = randint(0, output_size-1)

                        if output>=3 and output<=6:
                            temp = np.random.rand(1)
                            if temp>(episode/100):
                                pass
                                #output = 10

                        action = OutputToAction3(output)
                        print("random action:", output)

                    else:
                        # Choose an action by greedily from the Q-network
                        predicted = mainDQN.predict(acc_state, output_seq)
                        output = np.argmax(predicted)
                        action = OutputToAction3(output)

                        print("output:", output, "predicted:", predicted)
                        #print("output:", output)

                    # Get new state and reward from environment
                    for n in range(len(action)):
                        next_state, reward, done, info = env.step(action[n])
                        if done == True:
                            print("%dth:", n)
                            break
                    print("dms~~",reward)
                    state_buffer.append(next_state)
                    #next_state_buffer.append(next_state)

                    output_buffer.append(action)

                    prev_distance = distance
                    distance = info['distance']
                    got_distance = distance-prev_distance

                    past_score = score
                    score = info['score']
                    got_score = score-past_score

                    time = info['time']
                   
                    reward = got_score/50 + got_distance/30
                    
                    if reward>0:
                        print("reward:", reward)


                    if done: # Penalty
                        #time = info['time']

                        

                        reward += -1.0

                        if distance>=3000:
                            reward = 1
                        #reward += distance / 1000
                        print("last reward:", reward)



                    # the experience to our buffer
                    #print("state:", np.shape(state))
                    if step_count>=10:
                        acc_state = [state_buffer[-2-k] for k in range(5)]

                        state_buffer.popleft()
                        acc_state = np.reshape(acc_state, (input_size[0], input_size[1], input_size[2]))

                        acc_next_state = [state_buffer[-1-k] for k in range(5)]


                        acc_next_state = np.reshape(acc_next_state, (input_size[0], input_size[1], input_size[2]))

                        output_seq = [output_buffer[-2-k] for k in range(5)]
                        output_next_seq = [output_buffer[-1-k] for k in range(5)]
                        output_buffer.popleft()

                        replay_buffer.append((acc_state, output_seq, output_next_seq, output, reward, acc_next_state, done))
                        if replay_buffer[-1][6]: #if done==true?
                            for k in range(1, 5):
                                replay_buffer[-1 - k] = tuple(
                                    replay_buffer[-1 - k][0:4] + (-pow(0.9, k),) + replay_buffer[-1 - k][5:])
                        if replay_buffer[-1][4] >= 2.0 and replay_buffer[-1][6] == False:
                            for k in range(1, 5):
                                replay_buffer[-1 - k] = tuple(
                                    replay_buffer[-1 - k][0:4] + (pow(0.9, k),) + replay_buffer[-1 - k][5:])

                        #replay_buffer.append((state, action, reward, next_state, done))
                        if len(replay_buffer) > REPLAY_MEMORY:
                            replay_buffer.popleft()
                        acc_state = acc_next_state



                    state = next_state
                    step_count += 1
                    if step_count > 100000:   # Good enough. Let's move on
                        break

                    #if step_count==1:
                    #    replay_buffer.pop()



                print("Episode: {} steps: {}".format(episode, step_count))
                if step_count > 100000:
                    pass
                    # break

                if (episode+1) % 1 == 0: # train every 10 episode
                    # Get a random batch of experiences
                    for _ in range(50):
                        # Minibatch works better
                        if len(replay_buffer) >= 10:

                            sample_idx = random.sample(range(0, len(replay_buffer)), 10)

                            minibatch2 = []
                            for i in sample_idx:

                                minibatch2.append(replay_buffer[i])

                            l_rate =(1e-5 -1e-4)*(1/max_episodes)*episode + 1e-4
                            loss, _ = ddqn_replay_train(mainDQN, targetDQN, minibatch2, l_rate=l_rate)

                            print("Loss: ", loss, "l_rate:", l_rate)
                if (episode+1) % 2 == 0: # train every 10 episode
                    sess.run(copy_ops)
                    print("weights copied")

                if (episode + 1) % 100 == 0:  # train every 10 episode
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint_dir + 'model_notime_level2.ckpt')
                    print("model saved")
                    env.reset()

            # See our trained bot in action
            env2 = wrappers.Monitor(env, 'gym-results', force=True)

            for i in range(200):
                bot_play(mainDQN, env=env2)

            env2.close()

    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        with tf.Session(config=config) as sess:
            mainDQN = dqn.DQN(sess, input_size, output_size, name="main")
            targetDQN = dqn.DQN(sess, input_size, output_size, name="target")
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_dir + 'model_notime_level2.ckpt')
            for i in range(200):
                bot_play(mainDQN, env=env)

            env.close()

if __name__ == "__main__":
    main()



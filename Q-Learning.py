import numpy as np
import pandas as pd
import time


np.random.seed(2)  # 产生一组伪随机数，每次随机过程一样的
N_STATES = 20  # 宝藏的开始距离
ACTIONS = ['left', 'right']
EPSION = 0.99  # 90%的时候选取最优动作，1%选择随机动作
ALPHA = 0.1  # 学习效率
LAMBDA = 0.1  # 衰减度
MAX_EPISODES = 100  # 玩的回合数
FRESH_TIME = 0.1    # 走一步花的时间


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions,  # actions的名字
    )
    return table


# print(build_q_table(N_STATES, ACTIONS))

def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSION) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    global S_
    if A == 'right':  # right
        if S == N_STATES-2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:  # left
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def updata_env(S, episode, step_counter):
    env_list = ['-']*(N_STATES-1)+['T']  # our env
    if S == 'terminal':
        interaction = 'Episode%s:total_step = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction, end=''))
        time.sleep(2)
        print('\r                 ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction, end=''))
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 10  # 初始位置
        is_terminated = False
        updata_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA*q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.ix[S, A] += ALPHA*(q_target-q_predict)
            S = S_

            updata_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-Table:\n')
    print(q_table)
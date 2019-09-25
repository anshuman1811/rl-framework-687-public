import numpy as np
from rl687.environments.gridworld import Gridworld
import matplotlib.pyplot as plt
import time

def problemA():
    """
    Have the agent uniformly randomly select actions. Run 10,000 episodes.
    Report the mean, standard deviation, maximum, and minimum of the observed 
    discounted returns.
    """
    # setting random seed for reproducibility
    print ("Problem A")

    env = Gridworld()
    
    discounted_returns = []
    for episode in range(10000):
        # print (episode)
        discounted_return = 0.0
        while not env.isEnd:
            state = env.state
            action = np.random.choice([0,1,2,3])
            # print (state, action)
            actual_action, new_state, reward = env.step(action)
            # print (actual_action, new_state, reward)
            discounted_return += reward
            # print (t)
        env.reset()
        # print(time_step)
        discounted_returns.append(discounted_return)

    print ("Mean ", np.mean(discounted_returns))
    print ("Std Dev ", np.std(discounted_returns))
    print ("Max ", np.max(discounted_returns))
    print ("Min ", np.min(discounted_returns))

    return discounted_returns
    """
    Results:
        Mean  -9.123800743342608
        Std Dev  7.197419765658342
        Max  4.304672100000001
        Min  -46.72965105421902
    """

def problemB():
    """
    Run the optimal policy that you found for 10,000 episodes. Repor the 
    mean, standard deviation, maximum, and minimum of the observed 
    discounted returns
    """
    print ("Problem B")

    optimal_policy_actions = [1,1,1,1,2,  0,1,1,1,2,  0,2,-1,2,2,  0,3,-1,1,2,  0,3,1,1,-1]

    env = Gridworld()
    
    discounted_returns = []
    for t in range(10000):
        # print (t)
        discounted_return = 0.0
        while not env.isEnd:
            state = env.state
            action = optimal_policy_actions[state]
            # print (state, action)
            actual_action, new_state, reward = env.step(action)
            # print (actual_action, new_state, reward)
            discounted_return += reward
        discounted_returns.append(discounted_return)
        env.reset()

    print ("Mean ", np.mean(discounted_returns))
    print ("Std Dev ", np.std(discounted_returns))
    print ("Max ", np.max(discounted_returns))
    print ("Min ", np.min(discounted_returns))

    return discounted_returns
    # plt.hist(sorted(discounted_returns), density = True, cumulative=True, label='CDF',
    #      histtype='step', alpha=0.8, color='k')
    # plt.show()

    """
    Results:
        Mean  2.670185828345252
        Std Dev  3.2895320487842836
        Max  4.782969000000001
        Min  -25.05248903619
    """

def problemE():
    """
    Have the agent uniformly randomly select actions. Run 10,000 episodes.
    Report the mean, standard deviation, maximum, and minimum of the observed 
    discounted returns.
    """
    # setting random seed for reproducibility
    print ("Problem E")
    start_time = time.time()

    env = Gridworld(startState = 19)
    num_episodes = 1000000
    count_s19_22_given_s8_19 = 0
    for episode in range(num_episodes):
        # print (episode)
        time_step = 0
        while (not env.isEnd) and time_step<12:
            state = env.state
            if time_step == 11 and state == 22:
                count_s19_22_given_s8_19 += 1
            action = np.random.choice([0,1,2,3])
            env.step(action)
            time_step += 1
            # print (t)
        env.reset()
    print(count_s19_22_given_s8_19)
    Pr_s19_22_given_s8_19 = (count_s19_22_given_s8_19*1.0)/num_episodes

    end_time = time.time()
    print ("Estimate of Pr(S_8=19 | S_19 = 22) = ", Pr_s19_22_given_s8_19)
    print ("Execution time = ", end_time - start_time)
    """
        Estimate of Pr(S_8=19 | S_19 = 22) =  0.01873
    """

def quantile(discounted_returns):

    num_returns=len(discounted_returns)
    discounted_returns.sort()

    cdf = []
    print("CDF")

    for  r in sorted(set(discounted_returns)):
        cdf.append (len([i for i in discounted_returns if i<= r])/num_returns)    

    return cdf, sorted(set(discounted_returns))

def main():
    np.random.seed(123)

    returns_A = problemA()
    x,y = quantile(returns_A)
    plt.title("Quantile Function for Random Policy")
    plt.xlabel("Sample Fraction (CDF)")
    plt.ylabel("Quantile (Discounted Return)")
    plt.plot(x,y)
    plt.show()
    
    returns_B = problemB()
    x,y = quantile(returns_B)
    plt.plot(x,y)
    plt.title("Quantile Function for Optimal Policy")
    plt.xlabel("Sample Fraction (CDF)")
    plt.ylabel("Quantiles (Discounted Return)")
    plt.show()
    quantile(returns_B)

    # problemE()

main()
        

import numpy as np
from scipy import stats


class RewardFunction:

    # def __init__(self):

        # self.tir = 0
        # self.reward = []

    def calculate_reward(self, blood_glucose_level, reward_flag='absolute', bg_ref=108, action=None, blood_glucose_level_start=None):
        """
        Calculating rewards for the given blood glucose level
        """

        if reward_flag == 'binary':
            ''' Binary reward function'''
            low_bg = 70
            high_bg = 120

            if np.max(blood_glucose_level) < high_bg and np.min(blood_glucose_level) > low_bg:
                reward = 1
            else:
                reward = 0

        elif reward_flag == 'gamma':
            ''' gamma reward function within "safe" interval, negative polynomials outside boundary'''

            mode = 108
            low_bg = 72
            high_bg = 200
            x = blood_glucose_level
            reward = np.empty_like(blood_glucose_level)

            # Defining the parts of the reward function
            def gammaDist(x, mode, loc):
                a = 3.3
                theta = (mode-loc)/(a-1)
                dist = stats.gamma.pdf(x, a, loc=loc, scale=theta)
                dist_max = stats.gamma.pdf(mode, a, loc=loc, scale=theta)
                return dist/dist_max

            def posBg(x, shift):
                x = x-shift
                y = 0.05 - 0.005*x - (x**(2.3))/10000
                return y

            def negBg(x, shift):
                x = x-shift
                y = 0.02*x - x**(2)/1000
                return y

            # Select reward function part based on bg
            reward[x < low_bg] = negBg(x[x < low_bg], low_bg)
            reward[x > high_bg] = posBg(x[x > high_bg], high_bg)
            reward[(low_bg <= x)*(x <= high_bg)] = gammaDist(
                x[(low_bg <= x)*(x <= high_bg)], mode, low_bg)
            return reward

        elif reward_flag == 'binary_tight':
            ''' Tighter version of the binary reward function,
            the bounds are[-5, 5] around the optimal rate.
            '''
            # low_bg = bg_ref - 5
            # high_bg = bg_ref + 5

            low_bg = bg_ref - 10
            high_bg = bg_ref + 10

            if np.max(blood_glucose_level) < high_bg and np.min(blood_glucose_level) > low_bg:
                # reward = 200
                reward = 1
            else:
                reward = 0

        elif reward_flag == 'squared':
            ''' Squared cost function '''

            reward = - (blood_glucose_level - bg_ref)**2

        elif reward_flag == 'absolute':
            ''' Absolute cost function '''

            reward = - abs(blood_glucose_level - bg_ref)

        elif reward_flag == 'absolute_with_insulin':
            ''' Absolute cost with insulin constraint '''

            if action == None:
                action = [0, 0]

            # Parameters
            alpha = .7
            beta = 1 - alpha

            reward = - alpha*(abs(blood_glucose_level - bg_ref)
                              ) - beta * (abs(action[1]-action[0]))

        elif reward_flag == 'gaussian':
            ''' Gaussian reward function '''
            h = 30
            # h = 15
            # h = 10

            # reward = 200 * np.exp(-0.5 * (blood_glucose_level - bg_ref)**2 /h**2)
            reward = np.exp(-0.5 * (blood_glucose_level -
                                    bg_ref) ** 2 / h ** 2)

        elif reward_flag == 'gaussian_with_insulin':
            ''' Gaussian reward function '''
            h = 30
            # h = 15
            # h = 10
            alpha = .5

            bg_reward = np.exp(-0.5 * (blood_glucose_level - bg_ref)**2 / h**2)
            insulin_reward = -1/15 * action + 1

            # reward = 200 * alpha * bg_reward + (1 - alpha) * insulin_reward
            reward = alpha * bg_reward + (1 - alpha) * insulin_reward

        elif reward_flag == 'hovorka':
            ''' Sum of squared distances from target trajectory in Hovorka 2014 '''
            trgt = 6  # bg_ref/18? target bg is 6 mmol/l in Hovorka 2014

            # starting state added as input to calculate_reward
            y0 = blood_glucose_level_start/18

            # time until blood glucose has decreased to trgt+2 if y0 > trgt+2
            t1 = np.max((y0-trgt-2)/2, 0)

            # exponential half-time is 15 minutes (1/4h)
            r = 4*np.log(2)

            # target trajectory where starting bg is y0, time is in hours
            def y(t): return trgt + (y0-trgt-2*t)*(y0-2*t > trgt+2) + (y0-trgt-t1-t)*(
                trgt < y0-t1-t <= trgt+2) - (trgt-y0)*np.exp(-r*t)*(y0 < trgt)

            # how the index in blood_glucose_level relates to time in hours
            def t(i): return i/60

            reward = 0
            for i in range(len(blood_glucose_level)):
                reward = reward - (blood_glucose_level[i]/18 - y(t(i)))**2

        elif reward_flag == 'asy_tight':
            ''' Asymmetric tight reward function '''
            severe_low_bg = 54
            low_bg = 90
            high_bg = 180
            reward_aux = []

            # if np.min(blood_glucose_level) < severe_low_bg:
            for i in range(len(blood_glucose_level)):
                if blood_glucose_level[i] < severe_low_bg:
                    reward_aux.append(-100)
                    # reward_aux.append(-10)
                    # self.tir = 0
                # elif severe_low_bg <= blood_glucose_level < low_bg:
                elif severe_low_bg <= blood_glucose_level[i] < low_bg:
                    reward_aux.append(
                        np.exp((np.log(117.455)/low_bg) * blood_glucose_level[i]) - 117.455)
                    # reward_aux.append(np.exp((np.log(19.157) / low_bg) * blood_glucose_level[i]) - 19.157)
                    # self.tir = 0
                # elif low_bg <= blood_glucose_level < bg_ref:
                elif low_bg <= blood_glucose_level[i] < bg_ref:
                    reward_aux.append(((1 / 18) * blood_glucose_level[i] - 5))
                    # reward_aux.append(((1/36)*blood_glucose_level[i] - 2) + self.tir)
                    # self.tir = self.tir + 1
                # elif bg_ref <= blood_glucose_level <= high_bg:
                elif bg_ref <= blood_glucose_level[i] <= high_bg:
                    reward_aux.append(
                        ((-1 / 72) * blood_glucose_level[i] + (5 / 2)))
                    # reward_aux.append(((-1/72)*blood_glucose_level[i] + (5/2)) + self.tir)
                    # self.tir = self.tir + 1
                # else:
                elif high_bg < blood_glucose_level[i]:
                    reward_aux.append(0)
                    # reward_aux.append(-9)
                    # self.tir = 0

            reward = reward_aux

        elif reward_flag == 'asymmetric':
            ''' Asymmetric reward function '''
            severe_low_bg = 54
            low_bg = 72
            high_bg = 180
            reward_aux = []

            # if np.min(blood_glucose_level) < severe_low_bg:
            for i in range(len(blood_glucose_level)):
                if blood_glucose_level[i] < severe_low_bg:
                    reward_aux.append(-100)
                    # reward_aux.append(-10)
                    # self.tir = 0
                # elif severe_low_bg <= blood_glucose_level < low_bg:
                elif severe_low_bg <= blood_glucose_level[i] < low_bg:
                    reward_aux.append(
                        np.exp((np.log(140.9)/low_bg) * blood_glucose_level[i]) - 140.9)
                    # reward_aux.append(np.exp((np.log(19.157) / low_bg) * blood_glucose_level[i]) - 19.157)
                    # self.tir = 0
                # elif low_bg <= blood_glucose_level < bg_ref:
                elif low_bg <= blood_glucose_level[i] < bg_ref:
                    reward_aux.append(((1 / 36) * blood_glucose_level[i] - 2))
                    # reward_aux.append(((1/36)*blood_glucose_level[i] - 2) + self.tir)
                    # self.tir = self.tir + 1
                # elif bg_ref <= blood_glucose_level <= high_bg:
                elif bg_ref <= blood_glucose_level[i] <= high_bg:
                    reward_aux.append(
                        ((-1 / 72) * blood_glucose_level[i] + (5 / 2)))
                    # reward_aux.append(((-1/72)*blood_glucose_level[i] + (5/2)) + self.tir)
                    # self.tir = self.tir + 1
                # else:
                elif high_bg < blood_glucose_level[i]:
                    reward_aux.append(0)
                    # reward_aux.append(-9)
                    # self.tir = 0

            reward = reward_aux

        return reward

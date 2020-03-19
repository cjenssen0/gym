import numpy as np
from scipy import stats


class RewardFunction:

    def __init__(self):

        # self.tir = 0
        self.reward = np.zeros(30)

    def calculate_reward(self, blood_glucose_level, reward_flag='absolute', bg_ref=108, action=None, blood_glucose_level_start=None, tau_bg=1.):
        """
        Calculating rewards for the given blood glucose level
        """

        if reward_flag == 'gamma':
            ''' gamma reward function within "safe" interval, negative polynomials outside boundary'''
            mode = 108 / tau_bg
            low_bg = 72 / tau_bg
            high_bg = 200 / tau_bg
            x = blood_glucose_level / tau_bg
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

        elif reward_flag == 'gammaGauss':

            # Setting parameters
            a = 6.0
            low_bg = 72
            high_bg = 200

            alpha_low = 0.6
            alpha_high = 4
            sigma = np.sqrt(abs(bg_ref - low_bg))

            def gammaRev(x, a, mode, loc):
                # dist = stats.skewnorm(a, loc, scale)
                theta = (mode-loc)/(a-1)
                dist = stats.gamma(a, loc=loc, scale=theta)
                distMax = dist.pdf(mode)
                R = (dist.pdf(x) / distMax)
                return R

            def Gauss(x, mode, scale):
                dist = stats.norm(mode, scale)
                distMax = dist.pdf(mode)
                R = dist.pdf(x)/distMax - 1
                return R

            def rewGauss(self, x, a, mode, low_bg, high_bg, sigma):
                R = 0.
                x_I = x[(low_bg < x)*(x < high_bg)]
                x_low = x[x <= low_bg]
                x_high = x[x >= high_bg]

                if len(x_low) > 0:
                    R += sum(Gauss(x_low, low_bg, alpha_low*sigma))
                else:
                    R += 0.0
                if len(x_high) > 0:
                    R += sum(Gauss(x_high, high_bg, alpha_high*sigma))
                else:
                    R += 0.0
                R += sum(gammaRev(x_I, a, mode, low_bg))
                # self.reward[x <= low_bg] = Gauss(
                #     x[x <= low_bg], low_bg, alpha*sigma)
                # self.reward[x >= high_bg] = Gauss(
                #     x[x >= high_bg], high_bg, sigma)
                # self.reward[(low_bg < x)*(x < high_bg)
                #             ] = gammaRev(x_I, a, mode, low_bg)
                return R / len(blood_glucose_level)
            reward = rewGauss(self, blood_glucose_level, a,
                              bg_ref, low_bg, high_bg, sigma)
            return reward

        elif reward_flag == 'gaussian':
            ''' Gaussian reward function '''
            h = 30

            reward = np.exp(-0.5 * (blood_glucose_level -
                                    bg_ref) ** 2 / h ** 2)

        elif reward_flag == 'skewed_gaussian':
            ''' Skewed Gaussian reward function'''
            # set normalizing factor for states
            x = blood_glucose_level / tau_bg
            h = 42. / tau_bg
            loc = 91. / tau_bg
            a = 3.5
            reward = (1/0.016) * \
                stats.skewnorm.pdf(x, a, loc=loc, scale=h)
            reward[x < 72 / tau_bg] = -2.5
            reward[x > 200/tau_bg] = -2.

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

                elif severe_low_bg <= blood_glucose_level[i] < low_bg:
                    reward_aux.append(
                        np.exp((np.log(140.9)/low_bg) * blood_glucose_level[i]) - 140.9)

                elif low_bg <= blood_glucose_level[i] < bg_ref:
                    reward_aux.append(((1 / 36) * blood_glucose_level[i] - 2))

                elif bg_ref <= blood_glucose_level[i] <= high_bg:
                    reward_aux.append(
                        ((-1 / 72) * blood_glucose_level[i] + (5 / 2)))

                elif high_bg < blood_glucose_level[i]:
                    reward_aux.append(0)

            reward = reward_aux

        return reward

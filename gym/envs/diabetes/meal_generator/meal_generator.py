import numpy as np


def meal_generator(eating_time=1, premeal_bolus_time=0, meal_uncertainty_grams=20, no_meals=False, seed=None):
    '''Generates random meals

    Three meals per day
    '''

    # Fixing the random seed
    if seed is not None:
        np.random.seed(seed)

    # ==========================================
    # HACK to manually fix the random seed
    # ==========================================
    # np.random.seed(0)

    # meal_times = [round(np.random.uniform(330, 390)), round(np.random.uniform(690, 750)), round(np.random.uniform(1050, 1110))]
    # meal_amounts = [round(np.random.uniform(70, 90)), round(np.random.uniform(50, 70)), round(np.random.uniform(70, 90))]

    # Rounding off to nearest 5 minutes for simplicity
    # meal_times = [int(np.random.choice(np.linspace(330, 390, 5))), int(np.random.choice(np.linspace(690, 750, 5))), int(np.random.choice(np.linspace(1050, 1110, 5)))]
    # meal_amounts = [round(np.random.uniform(70, 90)), round(np.random.uniform(50, 70)), round(np.random.uniform(70, 90))]

    # Using the base-meals of Anas El Fathis work and adding +-30 mins to the times randomly
    meal_amounts = np.array([40, 80, 60, 30]) + np.random.uniform(-20, 20, 4)
    meal_times = np.array([8*60, 12*60, 18*60, 22*60]) + \
        np.random.choice(np.linspace(-30, 30, 3, dtype=int), 4)

    ### meal_amounts = np.array([40, 80, 60, 30])
    ### meal_times = np.array([8 * 60, 12 * 60, 18 * 60, 22 * 60])

    # meal_amounts = np.array([np.random.uniform(10, 100)])
    # meal_times = np.array([12 * 60]) + np.random.choice(np.linspace(-30, 30, 3, dtype=int))

    # Matlab meals
    # meal_amounts = np.array([53.7706299432407, 94.3178247049103, 73.8900695513650, 34.9425478714389])
    ###meal_amounts = np.array([39.75461948, 83.27555771, 45.05470101, 45.07282482])
    # meal_amounts = np.array([80, 0, 0, 0])
    # meal_amounts = np.array([0, 0, 0, 0])1110
    # meal_times = np.array([511, 691, 1111, 1291])
    # meal_times = np.array([120,  750, 1110, 1290])

    # Meal for estimating carb ratio -- a single meal
    # meal_amounts = np.array([40]) + np.random.uniform(-20, 20, 1)
    # meal_times = np.array([1*60]) #+ np.random.choice(np.linspace(-30, 30, 3, dtype=int), 1)
    # guessed_meal_amount = meal_amounts

    # Adding guessed meal amount
    # guessed_meal_amount = [round(np.random.uniform(meal_amounts[0]-20, meal_amounts[0]+20)), \
    # round(np.random.uniform(meal_amounts[1]-20, meal_amounts[1]+20)), round(np.random.uniform(meal_amounts[2]-20, meal_amounts[2]+20))]

    # guessed_meal_amount = meal_amounts + np.random.uniform(-, 20, 4)

    guessed_meal_amount = np.zeros_like(meal_amounts)
    for i in range(len(meal_amounts)):
        guessed_meal_amount[i] = meal_amounts[i] + \
            np.random.uniform(-meal_amounts[i]*.3, meal_amounts[i]*.3)

    ## guessed_meal_amount = np.array([42.00581209, 85.57842835, 41.38593027, 54.49588969])
    # guessed_meal_amount = np.array([40, 80, 60, 30])
    # guessed_meal_amount = np.array([100, 100, 100, 100])

    # eating_time = 1
    # premeal_bolus_time = 15
# time array([ 510,  690, 1050, 1290])
    # meals array([23.57606377, 79.53611082, 47.24522753, 14.68298779])
    # meal indicator array([20.61570784, 68.66430426, 46.07904586, 10.63029771])
    # Meals indicates the number of carbs taken at time t
    # meals = np.zeros(1440)
    meals = np.zeros(2160)

    # 'meal_indicator' indicates time of bolus - default 30 minutes before meal
    # meal_indicator = np.zeros(1440)
    meal_indicator = np.zeros(2160)

    for i in range(len(meal_times)):

        meals[meal_times[i]: meal_times[i] +
              eating_time] = meal_amounts[i]/eating_time * 1000 / 180
        # meal_indicator[meal_times[i] - premeal_bolus_time:meal_times[i]] = meal_amounts[i] * 1000 / 180

        # Changing to guessed meal amount
        meal_indicator[meal_times[i]-premeal_bolus_time:meal_times[i]-premeal_bolus_time +
                       eating_time] = guessed_meal_amount[i]/eating_time * 1000 / 180

    if no_meals:
        # meals = np.zeros(1440)
        # meal_indicator = np.zeros(1440)
        meals = np.zeros(2160)
        meal_indicator = np.zeros(2160)

    # Hack
    # meals = np.zeros(1440)
    # meal_indicator = np.zeros(1440)

    meal_bolus_indicator = meal_indicator

    return meals, meal_bolus_indicator

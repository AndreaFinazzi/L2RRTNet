import csv, pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

file_path = pathlib.Path(__file__).parent.absolute()
file_path_str = str(file_path)

num_problems = 10
num = 5000
eval_row_dim = num

def load_eval_data(folder):
    data_cost = []
    data_success = []

    filename = file_path_str + '/results/' + folder + '/eval_cost.csv' 
    f_cost = open(filename, 'rt', newline="")
    reader = csv.reader(f_cost, delimiter=',')
    for row in reader:
        row_data = map(float,row[0:eval_row_dim])
        data_cost.append(list(row_data))
    f_cost.close()

    filename = file_path_str + '/results/' + folder + '/eval_success.csv' 
    f_success = open(filename, 'rt', newline="")
    reader = csv.reader(f_success, delimiter=',')
    for row in reader:
        row_data = map(float,row[0:eval_row_dim])
        data_success.append(list(row_data))
    f_success.close()

    cost_rate = np.array(data_cost)
    success_rate = np.array(data_success)

    return success_rate, cost_rate

success_random, cost_random = load_eval_data('random')
success_025, cost_025 = load_eval_data('025')
success_050, cost_050 = load_eval_data('050')
success_075, cost_075 = load_eval_data('075')
success_100, cost_100 = load_eval_data('100')

success = {
    'random': success_random,
    '025': success_025,
    '050': success_050,
    '075': success_075,
    '100': success_100
}

cost = {
    'random': cost_random,
    '025': cost_025,
    '050': cost_050,
    '075': cost_075,
    '100': cost_100
}

# Get the best score for each sheet for each problem
best_costs = {}
for key, sheet in cost.items():
    best_costs[key] = [s for s in sheet[:,-1]]

best_costs_overall = []
for p_id in range(num_problems):
    costs = [best_cost_s[p_id] for best_cost_s in best_costs.values()]
    best_costs_overall.append(np.min(costs))

plt.bar(range(len(best_costs)), height=[np.mean(row) for row in list(best_costs.values())])
plt.xticks(range(len(best_costs)), best_costs.keys())
plt.show()


for (key, value) in success.items():
    plt.plot(value[0,:], label=key)

plt.legend()
plt.show()

for key, value in cost.items():
    # Normalize costs:
    scores = np.zeros((num_problems,num))
    for problem_id, cost_problem in enumerate(value):
        # best_score = cost_problem[-1]
        best_score = best_costs_overall[problem_id]
        worse_score = [cost for cost in cost_problem if cost!=np.inf]
        if len(worse_score)>0:
            worse_score=worse_score[0]
        else: 
            worse_score=np.inf
        scores[problem_id] = [(((cost-best_score)/(best_score))*(num_problems - success.get(key)[0, itr]) if cost!=np.inf else np.nan) for itr, cost in enumerate(cost_problem)]

    scores_total = np.ones((num))
    for col_id, element in enumerate(scores[0]):
        mean = np.nan
        col = [score for score in scores[:,col_id] if score >= 0]
        if len(col) > 0:
            # mean = np.sum(col)/success_rate[0,col_id]
            mean = np.mean(col) 
        scores_total[col_id] = mean

    axes = plt.gca()
    # axes.set_xlim([0,num])
    # axes.set_ylim([0.9,1.5])
    axes.relim()
    # update ax.viewLim using the new dataLim
    axes.autoscale_view()
    # plt.plot(scores_total, linestyle='None', marker='.')
    # plt.show()
    plt.plot(scores_total, label=key)
plt.legend()
plt.show()
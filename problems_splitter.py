import csv, pathlib

file_path = pathlib.Path(__file__).parent.absolute()
file_path_str = str(file_path)

img_res = 32
img_dim = img_res*img_res

x_dim = img_dim
y_dim = img_dim
u_dim = 2 # control effort
z_dim = 2
data_dim = 3*img_dim + u_dim # x_i, x_i+1, x_empty (to pass through obstacles), control input (x,y unit vector)

data_dim_problem = x_dim*3 # x_init, x_goal, x_empty
filename = file_path_str + '/data/problems.csv'
f = open(filename, 'rt')
reader = csv.reader(f, delimiter=',')
count = 0
data_list = []
for row in reader:
    row_data = map(float,row[0:data_dim])
    data_list.append(list(row_data))

num_problems = len(data_list)

print("Read ", num_problems, " problems")

data_train = []
data_test = []
data_eval = []

data_train = data_list[0:70]
data_test = data_list[70:90]
data_eval = data_list[90:]

filename_train = file_path_str + '/data/problems_train.csv' 
f_train = open(filename_train, 'w+', newline="")
writer_train = csv.writer(f_train, delimiter=',')

writer_train.writerows(data_train)
f_train.close()
    
filename_test = file_path_str + '/data/problems_test.csv' 
f_test = open(filename_test, 'w+', newline="")
writer_test = csv.writer(f_test, delimiter=',')

writer_test.writerows(data_test)
f_test.close()

filename_eval = file_path_str + '/data/problems_eval.csv' 
f_eval = open(filename_eval, 'w+', newline="")
writer_eval = csv.writer(f_eval, delimiter=',')

writer_eval.writerows(data_eval)
f_eval.close()
import matplotlib.pyplot as plt
import pdb, glob,os
import numpy as np

if not os.path.exists('img'):
    os.system('mkdir -p {}'.format('img'))

path_as_key = True
prefix = 'slurm-*.out'
data_file_list = glob.glob(prefix)
data_file_list.sort()
#data_file_list = ['slurm-10013800.out']
var_name = ['loss','pe', 'ge']
total_data = {}

import numpy
def smooth(x,window_len=11,window='flat'):
        if x.ndim != 1:
                raise "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=numpy.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=numpy.ones(window_len,'d')
        else:
                w=eval('numpy.'+window+'(window_len)')
        y=numpy.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

total_data = {}
for data_file in data_file_list:
    with open(data_file, 'r') as f:
        data_lines = f.readlines()
    file_path = ''.join(data_file.split('/')[:-1])

    for line_data, data in enumerate(data_lines):
        if 'checkpoint/' in data:
            file_id = float(data.split('temp3_')[-1].split('/')[0])
            if file_id not in total_data:
                total_data[file_id] = {}
        data_items = data.split('\t')
        for data_item in data_items:
            for var in var_name:
                if var in data_item:
                    try:
                        if var not in total_data[file_id]:
                            total_data[file_id][var] = []
                        data_val = data_item.split(var)[1].split('(')[0]
                        data_val = float(data_val)
                        total_data[file_id][var].append(data_val)
                    except:
                        continue

for var in var_name:
    #x = np.array(total_data['Epoch'])
    _, ax = plt.subplots(1,1, figsize=(8, 6))
    counter = 0
    for file_id in total_data:
        acc_data = np.array(total_data[file_id][var])
        acc_data = np.clip(acc_data, -500, 500)
        plt.plot(np.arange(acc_data.shape[0]), acc_data,
            label = file_id)
    plt.legend()
    plt.savefig('img/img_{}.png'.format(var))
    plt.close()
pdb.set_trace()

import os
import numpy as np
import logging
import time
import matplotlib.pyplot as plt

class Logger(object):
    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    def log(self, seed, tag, value):
        logdir = os.path.join(self.logdir, str(seed))
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        filename = os.path.join(logdir, tag+'.txt')
        with open(filename, 'a') as f:
            f.write(str(value) + ' ')

    def loadLogger(path='info', name=None):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="[ %(asctime)s  %(levelname)s] %(message)s",
                                      datefmt="%a %b %d %H:%M:%S %Y")
        sHandler = logging.StreamHandler()
        sHandler.setFormatter(formatter)

        logger.addHandler(sHandler)
        if path != "":
            work_dir = os.path.join(path,
                                    time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()))
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)

            fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
            fHandler.setLevel(logging.DEBUG)
            fHandler.setFormatter(formatter)

            logger.addHandler(fHandler)

        return logger

    @staticmethod
    def lineplot(x, y, label, dev=0):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rc('font', size=14)
        matplotlib.rc('axes.spines', top=False, right=False)
        matplotlib.rc('axes', grid=False)
        matplotlib.rc('axes', facecolor='white')
        plt.figure(figsize=(5, 5))
        _, ax = plt.subplots()
        ax.plot(x, y, lw=1, color='blue', alpha=1)
        # ax.fill_between(x, [y - dev for ], upper_CI, color='#539caf', alpha=0.4, label='95% CI')
        ax.set_xlabel('step')
        ax.set_ylabel(label)
        ax.legend(loc='best')
        plt.show()

    def plot(self, tag, seed_list):
        values = []
        for seed in seed_list:
            logdir = os.path.join(self.logdir, str(seed))
            filename = os.path.join(logdir, tag + '.txt')
            with open(filename, 'r') as f:
                s = f.readline()
                s = s.split(' ')[:-1]
                s = [float(i) for i in s]
                values.append(s)
        means = []
        for i in range(len(values[0])):
            sum = 0.0
            for j in range(len(values)):
                sum += values[j][i]
            mean = sum / len(values)
            means.append(mean)
        Logger.lineplot([i for i in range(len(values[0]))], means, tag)

    def tb_plot(self, tag, seed_list, tlogger):
        values = []
        for seed in seed_list:
            logdir = os.path.join(self.logdir, str(seed))
            filename = os.path.join(logdir, tag + '.txt')
            with open(filename, 'r') as f:
                s = f.readline()
                s = s.split(' ')[:-1]
                s = [float(i) for i in s]
                values.append(s)
        means = []
        for i in range(len(values[0])):
            sum = 0.0
            for j in range(len(values)):
                sum += values[j][i]
            mean = sum / len(values)
            means.append(mean)
        for i in range(len(means)):
            tlogger.log_scalar(tag, means[i], step=i)

    def get_values(self, tag, seed_list):
        values = []
        for seed in seed_list:
            logdir = os.path.join(self.logdir, str(seed))
            filename = os.path.join(logdir, tag + '.txt')
            with open(filename, 'r') as f:
                s = f.readline()
                s = s.split(' ')[:-1]
                s = [float(i) for i in s]
                s = s[:1200]
                values.append(self.smooth(s, 0.99))
                # bule:[0.38, 0.61, 1], [0.69, 0.8, 1]
                # green:[0, 0.73, 0.22], [0.5, 0.86, 0.61]
                # red:[0.97, 0.46, 0.43], [0.985, 0.73, 0.714]
        return values

    def mat_plot(self, value_list, label_list, color_list, rule_performance=-4.5):
        step = np.array([i for i in range(len(value_list[0][0]))])
        for i in range(len(value_list)):
            values_array = np.array(value_list[i])
            values_mean = values_array.mean(axis=0)
            values_std = values_array.std(axis=0)
            plt.plot(step, values_mean, color=color_list[i][0], label=label_list[i])
            plt.fill_between(step, values_mean-values_std, values_mean+values_std, color=color_list[i][1], alpha=0.5)
        plt.plot(step, [rule_performance for i in step], linestyle='--', label='rule')
        # plt.title('haha')
        plt.xlabel('Update number')
        plt.ylabel('Episode reward')
        plt.grid(linestyle='-.')
        axis = plt.gca()
        axis.spines['top'].set_color('gray')
        axis.spines['right'].set_color('gray')
        axis.spines['left'].set_color('gray')
        axis.spines['bottom'].set_color('gray')
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def smooth(scalars, weight): # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed
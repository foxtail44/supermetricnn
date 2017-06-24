import os
import csv
import torch
import torch.utils.data as data


def read_csv(path, bin_step):
    all_csv = []
    csv_files = os.listdir(path)

    for file in csv_files:
        with open(os.path.join(path, file)) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                all_csv += [[float(i) for i in row]]

    inputs = [i[0:10] for i in all_csv]
    outputs = [int(i[-1]/bin_step) for i in all_csv]

    return inputs, outputs


class SuperMetricParameters(object):
    def __init__(self, p):
        assert isinstance(p, dict)
        self.path_train = p["path_train"]
        self.path_eval = p["path_eval"]
        self.checkpoint_file = p["checkpoint_file"]
        self.model_best = p["model_best"]
        self.batch_size = p.get("batch_size", 10)
        self.learning_rate = p.get("learning_rate", 0.0001)
        self.epochs = p.get("epochs", 10000)
        self.print_freq = p.get("print_freq", 100)


class MetricDataLoader(data.Dataset):
    def __init__(self, csv_path, bin_step):
        self.inputs, self.outputs = read_csv(csv_path, bin_step=0.25)

    def __len__(self):
        return 2000
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.Tensor(self.inputs[item]).unsqueeze(1), self.outputs[item]


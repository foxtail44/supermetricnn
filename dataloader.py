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


class MetricDataLoader(data.Dataset):
    def __init__(self, csv_path, bin_step):
        self.inputs, self.outputs = read_csv(csv_path, bin_step=0.25)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return torch.Tensor(self.inputs[item]), self.outputs[item]


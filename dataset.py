import random


class Data:
    def __init__(self, data, label):
        self.data = data
        self.label = label


def read_data_set(filename):
    data_set = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = [x.strip() for x in line.split(',')]
            label = int(line[0])
            data = [int(x) for x in line[1].split(' ')]
            data_set.append(Data(data, label))
    return data_set


def split_data_set(data_set, train_part, validate_part, test_part):
    summed_parts = train_part + validate_part + test_part
    size = float(len(data_set))
    train_split = (train_part / summed_parts) * size
    validate_split = train_split + (validate_part / summed_parts) * size
    return data_set[:int(train_split)], data_set[int(train_split):int(validate_split)], data_set[int(validate_split):]


def enlarge_data_set(data_set, times, deviation):
    def get_randomized_data(data, deviation):
        return Data([d + random.uniform(-deviation, deviation) for d in data.data], data.label)

    size = len(data_set)
    for i in range(size):
        for j in range(times):
            data_set.append(get_randomized_data(data_set[i], deviation))

    random.shuffle(data_set)

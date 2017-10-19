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

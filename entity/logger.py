import os


class Logger(object):

    def __init__(self, header, batch=-1, autoStep=False, scroll=False):

        self.header = header
        self.titleCounts = None
        self.autoStep = autoStep
        self.scroll = scroll
        self.logBatch = False
        self.batch_id = 0
        self.batch = batch
        self.epoch = None
        self.initialize()
        self.spliter = ['---------' for _ in self.titleCounts]
        self.formatter = '{:^15}'.join(['|' for _ in self.titleCounts])

    def initialize(self):
        if self.batch > 0:
            self.epoch = 1
            self.logBatch = True
            self.header = ['epoch', 'batch'] + self.header
        self.titleCounts = range(len(self.header) + 1)

    def __enter__(self):
        self.printHeader()
        if not self.autoStep:
            self.step()
        return self

    def printHeader(self):
        self.__print(self.header, True)
        self.__print(self.spliter, True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log(self, message):
        self.__print(message)

    def log_yolos(self, yolos):
        for yolo in yolos:
            self.log(yolo.metrics.line())

    def step(self):
        self.batch_id += 1
        if not self.batch_id < self.batch + 1:
            self.epoch += 1
            self.batch_id = 1

    def __print(self, message, header=False):
        if self.scroll:
            os.system('clear')
            self.printHeader()
        if not header and self.logBatch:
            if self.autoStep:
                self.step()
            message = [self.epoch, self.batch_id] + message
        print(self.formatter.format(*message))


if __name__ == '__main__':
    head = ['name', 'gender', 'age']
    persons = [
        ['godme', 'male', '99'],
        ['judas', 'female', 88],
        ['godme', 'male', '99'],
        ['judas', 'female', 88],
        ['godme', 'male', '99'],
        ['judas', 'female', 88],
        ['godme', 'male', '99'],
        ['judas', 'female', 88],
        ['godme', 'male', '99'],
        ['judas', 'female', 88],
        ['godme', 'male', '99'],
        ['judas', 'female', 88],

    ]
    with Logger(head, 4, scroll=False) as logger:
        for person in persons:
            logger.log(person)
            logger.step()

from collections import OrderedDict
import os
import csv
import time
# import threading # for parallelism
# import queue # for parallelism
from multiprocessing import Process, Queue, cpu_count
from queue import Empty # to handle empty queues

import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, timing
from models import GCN

from train_and_save import train, test, write_results_to_file

class TestAndWriteProcess(Process):
    def __init__(self, queue, features, adj, idx_test, labels,records_to_write=50, records_filename="records.csv"):
        Process.__init__(self)
        self.queue = queue
        self.features = features
        self.adj = adj
        self.idx_test = idx_test
        self.labels = labels
        self.records_filename = records_filename
        self.records_to_write = records_to_write

    def run(self):
        count = 0
        while count < self.records_to_write:
            try:
                model = self.queue.get()
                # Testing
                metrics = test(model, self.features,
                               self.adj, self.idx_test, self.labels)
                params = model.state_dict()
                for k in params:
                    params[k] = params[k].reshape(-1).data.tolist()
                final_results = {**params, **metrics}
                # print("Model dequeued. Writing to file...")
                write_results_to_file(self.records_filename, final_results)
                count += 1
            except Empty:
                continue

class TrainingProcess(Process):
    def __init__(self, out_queue, features, adj, idx_train, labels):
        Process.__init__(self)
        self.out_queue = out_queue
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5
        self.features = features
        self.adj = adj
        self.idx_train = idx_train
        self.labels = labels

    def train(self):
        # Model and optimizer
        model = GCN(nfeat=self.features.shape[1],
                    nhid=self.hidden,
                    nclass=self.labels.max().item() + 1,
                    dropout=self.dropout)
        optimizer = optim.Adam(model.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        # Train model
        for _ in range(self.epochs):
            train(model, self.features, self.adj,
                  self.idx_train, self.labels, optimizer)
        return model

    def run(self):
        while True:
            model = self.train()
            self.out_queue.put(model)
            # print(f'Experiment {i+1} done')
        # print(f'Process is done populating buffer.')

def main():
    start_time = time.time()
    # Load data
    adj, features, labels, idx_train, idx_test = load_data()
    buffer = Queue()
    training_processes = []
    n_workers = cpu_count() * 10
    print(f"Spawning {n_workers} TrainingProcess's")
    for i in range(n_workers):
        t = TrainingProcess(buffer, features, adj, idx_train, labels)
        t.start()
        training_processes.append(t)

    print("Spawning TestAndWriteProcess")
    test_and_write = TestAndWriteProcess(buffer, features, adj, idx_test, labels)
    # test_and_write.daemon = False
    test_and_write.start()

    print("Joining on test_and_write")
    test_and_write.join()
    for t in training_processes:
        t.terminate()
    end_time = time.time()
    print("Done.")
    tasks_completed = test_and_write.records_to_write
    elapsed_time = end_time - start_time
    print(f'{tasks_completed} in {round(elapsed_time, 3)} seconds.')
    print(f'Task speed: {round(tasks_completed / elapsed_time, 3)} tasks/second.')

if __name__ == "__main__":
    main()

import json

import torch
import datasets
from get_model import get_model


class Server(object):

    def __init__(self, conf, eval_dataset):
        self.conf = conf

        self.global_model = get_model(self.conf["model_name"])

        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * (1 / self.conf["clients"])
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_train(self, train_loader):
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        self.global_model.train()
        for e in range(self.conf["local_epochs"]):

            for batch_id, batch in enumerate(train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()

                _, output = self.global_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

    # 模型评估
    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.global_model(data)
            # output = self.global_model(data)

            # sum up batch loss
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l


if __name__ == '__main__':
    with open("../utils/conf.json", 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"])

    server = Server(conf, eval_datasets)
    # print(server.global_model.state_dict().keys())

    a = torch.tensor([[[0., 1.], [0., 1.]], [[0., 1.], [0., -1.]]])
    b = torch.tensor([[[0., -1.], [0., 1.]], [[0., 1.], [0., 1.]]])
    cos = torch.nn.CosineSimilarity(dim=-1)
    print(cos(a, b))

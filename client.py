import logging

from torch.utils.data import DataLoader, Subset
import torch

from get_model import get_model


class Client(object):

    def __init__(self, conf, model, eval_dataset, train_dataset, non_iid, id=-1):
        self.client_id = id
        self.client_list = None
        self.cur_fit_layer = 0

        self.conf = conf
        self.local_model = get_model(self.conf["model_name"])
        self.local_model.load_state_dict(model.state_dict())

        sub_trainset: Subset = Subset(train_dataset, indices=non_iid)
        # print(non_iid)
        self.train_loader = DataLoader(sub_trainset, batch_size=conf["batch_size"], shuffle=False)
        # shuffle=True: 先将数据集随机置乱，再按照batchsize大小按顺序取

        self.eval_loader = DataLoader(eval_dataset, batch_size=conf["batch_size"], shuffle=False)

    def set_client(self, client_list):
        self.client_list = client_list

    def local_train(self, global_model):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                # print(target)
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()

                # FedAvg
                if self.conf["fedPKR"]==1:
                    feature, output = self.local_model(data)
                    feature_glo, _ = global_model(data)
                    loss1 = torch.nn.functional.cross_entropy(output, target)
                    loss2 = 0.0
                    self.cur_fit_layer %= self.conf["layers"]
                    loss2 += (feature[self.cur_fit_layer] - feature_glo[self.cur_fit_layer]).norm(2)
                    self.cur_fit_layer += 1
                    loss = loss1 + (self.conf["alpha"] / 2) * loss2
                    loss.backward()
                    optimizer.step()
                elif self.conf["fedPKR"] == 2:
                    feature, output = self.local_model(data)
                    feature_glo, _ = global_model(data)
                    loss1 = torch.nn.functional.cross_entropy(output, target)
                    loss2 = 0.0
                    self.cur_fit_layer = self.conf["layers"]-1
                    loss2 += (feature[self.cur_fit_layer] - feature_glo[self.cur_fit_layer]).norm(2)
                    loss = loss1 + (self.conf["alpha"] / 2) * loss2
                    loss.backward()
                    optimizer.step()
                else:
                    _, output = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

                # for i in range(len(feature)-1):
                #     loss2 += (feature[i] - feature_glo[i]).norm(2)
                # loss3 = torch.nn.functional.kl_div(feature[-1], feature_glo[-1])



                # My Scheme2 拟合本地模型
                # _, output = self.local_model(data)
                # loss1 = torch.nn.functional.cross_entropy(output, target)
                # loss2 = 0.0
                # for c in self.client_list:
                #     if c.client_id != self.client_id:
                #         _, other_output = c.local_model(data)
                #         loss2 += torch.nn.functional.kl_div(output, other_output)
                # loss = loss1+(self.conf["alpha"]/2)*loss2
                # loss.backward()
                # optimizer.step()

            print("Client {} Epoch {} done.".format(self.client_id, e))

            # logging.info("Client {} Epoch {} acc {}".format(self.client_id, e, self.model_eval()))
        # logging.info('\n')
        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return diff

    def model_eval(self):
        self.local_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.local_model(data)
            total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        total_l = total_loss / dataset_size

        return acc, total_l

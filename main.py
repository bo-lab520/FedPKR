import copy
import json
import logging

from server import Server
from client import *
import datasets

if __name__ == '__main__':

    with open("conf.json", 'r') as f:
        conf = json.load(f)

    if conf["fedPKR"] == 1:
        filename = "fedPKR," + conf["type"] + "," + "n=" + str(conf["clients"]) + ",alpha=" + str(
            conf["dirichlet_alpha"]) + ",mu=" + str(conf["mu"]) + ",fix,shuffle" + ".log"
    elif conf["fedPKR"] == 2:
        filename = "fedNPKR," + conf["type"] + "," + "n=" + str(conf["clients"]) + ",alpha=" + str(
            conf["dirichlet_alpha"]) + ",mu=" + str(conf["mu"]) + ",fix" + ".log"
    else:
        filename = "fedavg," + conf["type"] + "," + "n=" + str(conf["clients"]) + ",alpha=" + str(
            conf["dirichlet_alpha"]) + ",shuffle" + ".log"
    logging.basicConfig(level=logging.INFO,
                        filename="../log/" + filename,
                        filemode='w')

    train_datasets, eval_datasets = datasets.get_dataset("../data/", conf["type"])

    server = Server(conf, eval_datasets)

    clients = []

    # non-IID数据
    client_idx = datasets.dirichlet_nonIID_data(train_datasets, conf)

    for c in range(conf["clients"]):
        clients.append(Client(conf, server.global_model, eval_datasets, train_datasets, client_idx[c + 1], c + 1))

    for c in clients:
        c.set_client(clients)

    all_acc = []
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)

        if e == 40:
            conf["lr"] *= 0.1
        if e == 80:
            conf["lr"] *= 0.1

        # 调整alpha，逐渐减小
        # conf["alpha"] = conf["mu"] * (conf["global_epochs"] - (int(e / conf["layers"]) + 1) * conf["layers"]) / conf["global_epochs"]
        conf["alpha"] = conf["mu"]

        candidates = clients

        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            diff = c.local_train(server.global_model)
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)

        acc, loss = server.model_eval()
        all_acc.append(acc)
        print("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
        logging.info("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    print(all_acc)
    logging.info(str(all_acc))

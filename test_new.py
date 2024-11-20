import torch
import torch.nn as nn
import time
from util.time import *
from util.env import *
from util.data import *
from util.preprocess_new import *


def test(model, test_dataloader, phy_adj, net_adj, mul_adj):
    loss_func = nn.L1Loss()
    device = get_device()

    phy_test_loss_list = []
    net_test_loss_list = []
    now = time.time()

    test_labels_list = []
    test_labels_list_net = []

    t_test_phy_predicted_list = []
    t_test_phy_ground_list = []
    t_test_net_predicted_list = []
    t_test_net_ground_list = []
    t_test_labels_list = []
    

    test_len = len(test_dataloader)
    
    model.eval()
    i = 0
    acu_loss = 0
    for x, py, ny, labels, edge_index in test_dataloader:
        x, py, ny, labels, edge_index = [item.to(device).float() for item in [x, py, ny, labels, edge_index]]
        with torch.no_grad():
            outs, learned_graph, phy_graph, net_graph = model(x, phy_adj, net_adj, mul_adj)

            predicted_phy = outs[0]
            predicted_phy = predicted_phy.float().to(device)
            predicted_net = outs[1]
            predicted_net = predicted_net.float().to(device)
            loss_phy = loss_func(predicted_phy, py)
            loss_net = loss_func(predicted_net, ny)

            labels = labels.unsqueeze(1).repeat(1, predicted_phy.shape[1])

            if len(t_test_phy_predicted_list) <= 0:
                t_test_phy_predicted_list = predicted_phy
                t_test_phy_ground_list = py
                t_test_labels_list = labels
            else:
                t_test_phy_predicted_list = torch.cat((t_test_phy_predicted_list, predicted_phy), dim=0)
                t_test_phy_ground_list = torch.cat((t_test_phy_ground_list, py), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

            if len(t_test_net_predicted_list) <= 0:
                t_test_net_predicted_list = predicted_net
                t_test_net_ground_list = ny
            else:
                t_test_net_predicted_list = torch.cat((t_test_net_predicted_list, predicted_net), dim=0)
                t_test_net_ground_list = torch.cat((t_test_net_ground_list, ny), dim=0)
        
        phy_test_loss_list.append(loss_phy.item())
        acu_loss += loss_phy.item()

        net_test_loss_list.append(loss_net.item())
        acu_loss += loss_net.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    test_phy_predicted_list = t_test_phy_predicted_list.tolist()        
    test_phy_ground_list = t_test_phy_ground_list.tolist()      
    test_net_predicted_list = t_test_net_predicted_list.tolist()        
    test_net_ground_list = t_test_net_ground_list.tolist()    
    test_labels_list = t_test_labels_list.tolist()   
    test_labels_list_net = t_test_labels_list.repeat(1,3).tolist()
    
    phy_avg_loss = sum(phy_test_loss_list)/len(phy_test_loss_list)
    net_avg_loss = sum(net_test_loss_list)/len(net_test_loss_list)

    avg_loss = []
    avg_loss.append(phy_avg_loss)
    avg_loss.append(net_avg_loss)


    phy_list = [test_phy_predicted_list, test_phy_ground_list, test_labels_list]
    net_list = [test_net_predicted_list, test_net_ground_list, test_labels_list_net]

    return avg_loss, phy_list, net_list 





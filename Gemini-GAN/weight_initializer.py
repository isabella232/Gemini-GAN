from torch import nn
import torch 

class Initializer:
    def __init__(self):
        pass

    @staticmethod
    def initialize(model, initialization, **kwargs):

        def weights_init(m):
            '''
            Usage:
                model = Model()
                model.apply(weight_init)
            '''
            if isinstance(m, nn.Conv1d):
                torch.nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose1d):
                torch.nn.init.normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.ConvTranspose3d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
                torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
                torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
                torch.nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.normal_(m.bias.data)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        torch.nn.init.normal_(param.data)
            elif isinstance(m, nn.LSTMCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        v.normal_(param.data)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        torch.nn.init.normal_(param.data)
            elif isinstance(m, nn.GRUCell):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.orthogonal_(param.data)
                    else:
                        torch.nn.init.normal_(param.data)

        model.apply(weights_init)

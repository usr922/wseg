import torch
from sklearn.metrics import precision_recall_fscore_support as score
from torch.nn import functional as F

from util import pyutils


def validate(model, data_loader, epoch, args):
    print('\nvalidating ... ', flush=True, end='')
    val_loss_meter = pyutils.AverageMeter('loss')
    model.eval()

    corrects = torch.tensor([0 for _ in range(20)])
    tot_cnt = 0.0
    with torch.no_grad():
        y_true = list()
        y_pred = list()
        for i, pack in enumerate(data_loader):
            _, img, label = pack
            label = label.cuda(non_blocking=True)
            x, _, _ = model(img)

            x = x[:, :-1]

            loss = F.multilabel_soft_margin_loss(x, label)
            val_loss_meter.add({'loss': loss.item()})

            x_sig = torch.sigmoid(x)
            corrects += torch.round(x_sig).eq(label).sum(0).cpu()

            y_true.append(label.cpu())
            y_pred.append(torch.round(x_sig).cpu())

            tot_cnt += label.size(0)

        y_true = torch.cat(y_true, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        corrects = corrects.float() / tot_cnt
        mean_acc = torch.mean(corrects).item() * 100.0

        precision, recall, f1, _ = score(y_true.numpy(), y_pred.numpy(), average=None)

        precision = torch.tensor(precision).float()
        recall = torch.tensor(recall).float()
        f1 = torch.tensor(f1).float()
        mean_precision = precision.mean()
        mean_recall = recall.mean()
        mean_f1 = f1.mean()

    model.train()
    print('loss:', val_loss_meter.pop('loss'))
    print("Epoch({:03d})\t".format(epoch))
    print("MeanACC: {:.2f}\t".format(mean_acc))
    print("MeanPRE: {:.4f}\t".format(mean_precision))
    print("MeanREC: {:.4f}\t".format(mean_recall))
    print("MeanF1: {:.4f}\t".format(mean_f1))
    print("{:10s}: {}\t".format("ClassACC", " ".join(["{:.3f}".format(x) for x in corrects.cpu().numpy()])))
    print("{:10s}: {}\t".format("PRECISION", " ".join(["{:.3f}".format(x) for x in precision.cpu().numpy()])))
    print("{:10s}: {}\t".format("RECALL", " ".join(["{:.3f}".format(x) for x in recall.cpu().numpy()])))
    print("{:10s}: {}\n".format("F1", " ".join(["{:.3f}".format(x) for x in f1.cpu().numpy()])))
    return

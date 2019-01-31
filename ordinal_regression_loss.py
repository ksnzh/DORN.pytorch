import torch
from torch.autograd import Variable


class OrdinalRegressionLoss(torch.nn.Module):
    def __init__(self, K):
        super(OrdinalRegressionLoss, self).__init__()
        self.CrossEntropy = torch.nn.CrossEntropyLoss(size_average=False, ignore_index=-100)
        self.K = K

    def forward(self, feature, target):
        """
        :param feature: N*2K*H*W
        :param target: N*H*W
        :return:
        """
        loss = 0
        for k in xrange(self.K):
            # calculate the (2k,2k+2) channel
            mask_lt = torch.gt(target, k)   # k-th: the current k is less than the gt label
            mask_gt = torch.le(target, k)   #

            target_lt = torch.ones_like(target)
            target_lt[mask_gt.data] = -100

            target_gt = torch.zeros_like(target)
            target_gt[mask_lt.data] = -100

            mask_lt = mask_lt.type(torch.cuda.FloatTensor).unsqueeze(1)
            count_lt = torch.sum(mask_lt)
            count_gt = mask_lt.size(0) * mask_lt.size(2) * mask_lt.size(3) - count_lt

            # feature_k = feature.narrow(1, 2*k, 2)

            loss += self.CrossEntropy(feature.narrow(1, 2*k, 2), target_lt) / (count_lt + 1)
            loss += self.CrossEntropy(feature.narrow(1, 2*k, 2), target_gt) / (count_gt + 1)

        return loss

if __name__ == "__main__":
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)

    ordinal_loss = OrdinalRegressionLoss(K=2)
    feature = torch.randn(8, 4, 28, 28).cuda()
    feature = Variable(feature, requires_grad=True)
    target = torch.ones(8, 28, 28).type(torch.cuda.LongTensor).cuda()
    target = Variable(target)
    loss = ordinal_loss(feature, target)
    print loss
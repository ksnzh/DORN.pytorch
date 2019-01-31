import torch


def dorn_decode(net, input, K):
    output = net.forward(input)
    # output = F.upsample(output, scale_factor=4, mode='bilinear')
    N,_,H,W = input.size()
    decode = torch.zeros(N,1,H,W).type(torch.cuda.LongTensor)
    for k in xrange(K):
        decode += output.narrow(1, 2*k, 2).data.max(1, keepdim=True)[1]

    return decode
import torch
from torch import nn

"""
Channel Self-Attention Module: dubbed as CSA

SE Module:
    X[b,c,h,w] -> AvgPool:[b,c,1,1] -> FC:[b,c/r,1,1] -> ReLU -> FC:[b,c,1,1] -> Sigmoid:X_scale[b,c,1,1]
    X = X*X_scale + X

CSA Module:
    1) usual conv: X[b,c,h,w] -> Conv(k=3,s=1,p=1)+LN+GeLU:[b,c,h,w](X_res)
    2) channel order:
    AvgPool -> Conv(c->c*r,k=1) -> GeLU -> Conv(c*r->c,k=1)-> Sigmoid -> Channel_Order(from small to high)
    3) exchange channel SA:
    X_res = {X_resC1,...,X_resCc} -> {X_resCi,...,X_resCj} according to Channel_Order

    X_res += X_res @ X_exc
"""
class CSA(nn.Module):
    def __init__(self, inc=512, kernel_size=3, ratio=0.25):
        super(CSA, self).__init__()
        self.inconv = nn.Conv3d(inc, inc, kernel_size, 1, 1)
        self.innorm = nn.InstanceNorm3d(inc)
        # self.gelu = nn.GELU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.avg = nn.AdaptiveAvgPool3d(1)
        self.ch_order = nn.Sequential(
            nn.Linear(inc, int(inc*ratio)),
            # nn.GELU(),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(int(inc*ratio), inc),
            # nn.GELU(),
            nn.Sigmoid()
        )

    def forward(self, x:torch.Tensor):
        b,c,d,h,w = x.size()

        # x = self.inconv(x)
        # x = x.permute(0,2,3,4,1)
        # x_res = self.gelu(self.innorm(x)).permute(0,4,1,2,3)
        # del x
        x = self.inconv(x)
        x = self.lrelu(self.innorm(x))
        x_res = x
        ch_order = torch.argsort(self.ch_order(self.avg(x).view(b,c)))
        
        return x_res + self.exchange3(x, ch_order)
    
    # def forward(self, x:torch.Tensor):
    #     b,c,d,h,w = x.size()

    #     # x = self.inconv(x)
    #     # x = x.permute(0,2,3,4,1)
    #     # x_res = self.gelu(self.innorm(x)).permute(0,4,1,2,3)
    #     # del x

    #     ch_order = torch.argsort(self.ch_order(self.avg(x).view(b,c)))
        
    #     return x + x * self.exchange3(x, ch_order)

    @staticmethod
    def exchange_channel3d(x: torch.Tensor, channel_order: torch.Tensor) -> torch.Tensor:
    
        assert len(x.size()) == 5, f"only support 3d tensor, [b,c,d,h,w]"
        b,c,d,h,w = x.size()
        assert len(set(channel_order)) == len(channel_order) and all(i in channel_order for i in range(c)), \
            f"check channel_order which must follow the rules:"\
            f"consist of all channel order, e.g. for 8 channel, valid channel_order is like [0,3,4,1,2,6,5,7]"
        
        new_x = []
        for i in range(b):
            x1 = x[i,...].unsqueeze(0).reshape(1,-1)
            index = torch.arange(c*d*h*w)
            index = torch.cat([index[u*d*h*w:(u+1)*d*h*w] for u in channel_order[i]], dim=0).unsqueeze(0).to(x.device)
            x1 = x1.gather(dim=-1,index=index)
            x1 = x1.view(1,c,d,h,w)
            new_x.append(x1)

        return torch.vstack(new_x)
    
    @staticmethod
    def exchange(x: torch.Tensor, channel_order: torch.Tensor):
        b,c,d,h,w = x.size()
        new_x = []
        for batch in range(b):
            batch_order = channel_order[batch]
            batch_order = batch_order.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(x[0].size())
            new_x.append(x[batch].gather(0, batch_order).unsqueeze(0))
        return torch.vstack(new_x)

    @staticmethod
    def exchange3(x: torch.Tensor, channel_order: torch.Tensor):
        b,c,d,h,w = x.size()
        new_x = []
        for batch in range(b):
            batch_order = channel_order[batch]
            new_x.append(x[batch][batch_order].unsqueeze(0))
        return torch.vstack(new_x)
        
if __name__ == "__main__":
    x = torch.randn(2,16,8,8,8)
    csa = CSA(16)
    channel_order = torch.tensor([[1,2,3,4,5,6,8,7,9,0,10,12,11,15,14,13],[1,2,3,4,5,6,8,7,9,0,10,12,11,15,14,13]])

    x1 = csa.exchange_channel3d(x.clone(),channel_order)
    x2 = csa.exchange(x.clone(),channel_order)
    x3 = csa.exchange3(x.clone(),channel_order)
    print(torch.all(x1==x2), torch.all(x2==x3))
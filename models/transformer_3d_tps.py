from utils.util import *


class get_model(nn.Module):
    def __init__(self, d_model=128, channel=3,npoint=512):
        super(get_model, self).__init__()
        self.transformer = nn.Transformer(d_model, num_encoder_layers=4, num_decoder_layers=4)
        # self.stn=STNkd(d_model)
        self.pointnet = PointNetEncoder(channel, d_model)
        self.pointnetDecoder = PointNetDecoder( d_model)

        self.fc1 = nn.Linear(32*npoint, 243)
        self.fc2 = nn.Linear(243, 162)
        self.fc3 = nn.Linear(162, 81)
        self.rigid_3d=gen_3d_std_rigid()

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor):#->(torch.Tensor,torch.Tensor):
        #### Do not need z_order any more
        # temp1, temp2 = Z_order_sorting.apply(encoder_input[:, :, :3], encoder_input[:, :, 3:6])
        # encoder_input = torch.cat([temp1, temp2], -1)
        # temp1, temp2 = Z_order_sorting.apply(decoder_input[:, :, :3], decoder_input[:, :, 3:6])
        # decoder_input = torch.cat([temp1, temp2], -1)
        encoder_input, decoder_input = encoder_input.permute(0, 2, 1), decoder_input.permute(0, 2, 1)   #[B,C,N]
        embed_input = self.pointnet(encoder_input)
        embed_output = self.pointnet(decoder_input)
        embed_input, embed_output = embed_input.permute(2, 0, 1), embed_output.permute(2, 0, 1)  # [N, B, d_model]
        transformer_out = self.transformer(embed_input, embed_output)  # [N, B ,d_model]
        transformer_out = transformer_out.permute(1, 0, 2)  # [B, N, d_model]

        warpped_feat = transformer_out
        #这里直接做全连接的话参数量太过巨大了，所以先降维吧
        warpped_feat = self.pointnetDecoder(warpped_feat.permute(0,2,1))    #[B,32,N]
        warpped_feat = torch.flatten(warpped_feat, 1, -1)
        warpped_feat = torch.nn.functional.leaky_relu(self.fc1(warpped_feat))
        warpped_feat = torch.nn.functional.leaky_relu(self.fc2(warpped_feat))
        new_ctl = self.fc3(warpped_feat)
        B,C,N=decoder_input.size()
        new_ctl=torch.reshape(new_ctl,[B,27,3])#.cpu()
        print(f"new_ctl: min={new_ctl.min().item():.4f}, max={new_ctl.max().item():.4f}")
        #decoder_input=decoder_input.cpu()
        warped=torch.zeros([B,N,3], device=decoder_input.device)
        for aosidf in range(B):
            warped[aosidf]=tps3d(self.rigid_3d,new_ctl[aosidf],decoder_input[aosidf].permute(1,0))

       # loss1 = chamfer_loss(encoder_input[:, :3, :].cpu(), warped.permute(0,2,1), ps=N)

        return warped.permute(0, 2, 1)
    



if __name__ == '__main__':
    model = get_model(128, 3,512)
    a = torch.rand((2, 512, 3))
    b = torch.rand((2, 512, 3))
    out = model(a, b)
    print(out.shape)

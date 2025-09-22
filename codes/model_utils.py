import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision

class Variational1DBlock(nn.Module):
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        features:list = [1024, 512],
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        in_f = input_dim
        for out_f in features:
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.BatchNorm1d(out_f))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_f = out_f
        self.feature_extractor = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(out_f, output_dim)
        self.fc_var = nn.Linear(out_f, output_dim)
    def forward(self, x):
        if isinstance(x, dict):
            z = F.relu(self.feature_extractor(x['input']))
        else:
            z = F.relu(self.feature_extractor(x))
        mu = self.fc_mu(z)
        std = torch.exp(0.5 * self.fc_var(z))
        eps = torch.randn_like(std)
        return eps * std + mu  

class Linear1DBlock(nn.Module):
    def __init__(
        self,
        input_dim:int,
        output_dim:int,
        features:list = [1024, 512],
        norm_out:bool = False,
        relu_out:bool = False,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = []
        in_f = input_dim
        for out_f in features:
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.BatchNorm1d(out_f))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(p=dropout_prob))
            in_f = out_f
        layers.append(nn.Linear(out_f, output_dim))
        if norm_out:
            layers.append(nn.BatchNorm1d(output_dim))
        if relu_out:
            layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        if isinstance(x, dict):
            return self.layers(x['input'])
        else:
            return self.layers(x)

class ImgFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels:int = 5,
        out_features:int = 512,
        model_arch: str = 'densenet121',
        pretrained: bool = False,
    ):
        super().__init__()
        self.out_features = out_features
        f_extractor_2d = torchvision.models.__dict__[model_arch](pretrained = pretrained).features
        #####
        # 2D feature extractor
        #####
        # Fix first module when in_channel is not 3
        if in_channels != 3:
            old_conv = f_extractor_2d[0]
            if model_arch.startswith('vgg'):
                new_conv = nn.Conv2d(in_channels, 64, 3, 1, 1)
            elif model_arch.startswith('densenet'):
                new_conv = nn.Conv2d(in_channels, 64, 7, 2, 1, bias = False)
            new_state_dict = old_conv.state_dict()
            old_weight = old_conv.weight.data
            new_weight = torch.cat([old_weight.repeat(1, in_channels // 3, 1, 1), old_weight[:, :in_channels % 3]], dim = 1)
            new_state_dict['weight'] = new_weight
            new_conv.load_state_dict(new_state_dict)
            f_extractor_2d[0] = new_conv
        #####
        # 1D feature extractor following by 2D feature extractor
        #####
        if model_arch.startswith('vgg'):
            inc = f_extractor_2d[-3].out_channels
            f_extractor_1d = nn.Sequential(
                nn.Linear(inc, 1000, bias = True),
                nn.ReLU(),
                nn.BatchNorm1d(1000),
                nn.Linear(1000, out_features)
            )
        elif model_arch.startswith('densenet'):
            inc = f_extractor_2d[-1].num_features
            f_extractor_1d = nn.Sequential(
                torchvision.models.__dict__[model_arch](pretrained = pretrained).classifier,
                nn.ReLU(),
                nn.BatchNorm1d(1000),
                nn.Linear(1000, out_features)
            )
        self.f_extractor_2d = f_extractor_2d
        self.f_extractor_1d = f_extractor_1d
    def forward(self, x):
        if isinstance(x, dict):
            features = self.f_extractor_2d(x['image'])
        else:
            features = self.f_extractor_2d(x)
        out = F.relu(features, inplace = True)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.f_extractor_1d(out)
        return out
    
class VariationalImgFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels:int = 5,
        out_features:int = 512,
        model_arch: str = 'densenet121',
        pretrained: bool = False,
    ):
        super().__init__()
        self.out_features = out_features
        f_extractor_2d = torchvision.models.__dict__[model_arch](pretrained = pretrained).features
        #####
        # 2D feature extractor
        #####
        # Fix first module when in_channel is not 3
        if in_channels != 3:
            old_conv = f_extractor_2d[0]
            if model_arch.startswith('vgg'):
                new_conv = nn.Conv2d(in_channels, 64, 3, 1, 1)
            elif model_arch.startswith('densenet'):
                new_conv = nn.Conv2d(in_channels, 64, 7, 2, 1, bias = False)
            new_state_dict = old_conv.state_dict()
            old_weight = old_conv.weight.data
            new_weight = torch.cat([old_weight.repeat(1, in_channels // 3, 1, 1), old_weight[:, :in_channels % 3]], dim = 1)
            new_state_dict['weight'] = new_weight
            new_conv.load_state_dict(new_state_dict)
            f_extractor_2d[0] = new_conv
        #####
        # 1D feature extractor following by 2D feature extractor
        #####
        if model_arch.startswith('vgg'):
            inc = f_extractor_2d[-3].out_channels
            f_extractor_1d = nn.Sequential(
                nn.Linear(inc, 1000, bias = True),
                nn.ReLU(),
                nn.BatchNorm1d(1000),
                nn.Linear(1000, out_features)
            )
        elif model_arch.startswith('densenet'):
            inc = f_extractor_2d[-1].num_features
            f_extractor_1d = nn.Sequential(
                torchvision.models.__dict__[model_arch](pretrained = pretrained).classifier,
                nn.ReLU(),
                nn.BatchNorm1d(1000),
                nn.Linear(1000, out_features)
            )
        self.f_extractor_2d = f_extractor_2d
        self.f_extractor_1d_mu = copy.deepcopy(f_extractor_1d)
        self.f_extractor_1d_var = copy.deepcopy(f_extractor_1d)
    def forward(self, x):
        if isinstance(x, dict):
            features = self.f_extractor_2d(x['image'])
        else:
            features = self.f_extractor_2d(x)
        out = F.relu(features, inplace = True)
        out = F.adaptive_avg_pool2d(out, (1,1))
        z = torch.flatten(out, 1)
        mu = self.f_extractor_1d_mu(z)
        std = torch.exp(0.5 * self.f_extractor_1d_var(z))
        eps = torch.randn_like(std)
        return eps * std + mu

class CompositeEncoder(nn.Module):
    def __init__(
        self,
        module_1d:nn.Module,
        module_2d:nn.Module,
        features:list,
        latent_dim: int = 512,
        norm_out:bool = False,
        relu_out:bool = False
    ):
        super().__init__()
        self.module_1d = module_1d
        self.module_2d = module_2d
        layers = []
        features_1d = module_1d.output_dim
        features_2d = module_2d.out_features
        in_f = features_1d + features_2d
        for out_f in features:
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.BatchNorm1d(out_f))
            layers.append(nn.ReLU())
            in_f = out_f
        layers.append(nn.Linear(out_f, latent_dim))
        if norm_out:
            layers.append(nn.BatchNorm1d(latent_dim))
        if relu_out:
            layers.append(nn.ReLU())
        self.merge_1d = nn.Sequential(*layers)
    def forward(self, x:dict):
        z_exp = self.module_1d(x['input'])
        z_img = self.module_2d(x['image'])
        return self.merge_1d(torch.cat([z_exp, z_img], dim = 1))

class VariationalCompositeEncoder(nn.Module):
    def __init__(
        self,
        module_1d:nn.Module,
        module_2d:nn.Module,
        features:list,
        latent_dim: int = 512,
        norm_out:bool = False,
        relu_out:bool = False
    ):
        super().__init__()
        self.module_1d = module_1d
        self.module_2d = module_2d
        layers = []
        features_1d = module_1d.output_dim
        features_2d = module_2d.out_features
        in_f = features_1d + features_2d
        for out_f in features:
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nn.BatchNorm1d(out_f))
            layers.append(nn.ReLU())
            in_f = out_f
        layers.append(nn.Linear(out_f, latent_dim))
        if norm_out:
            layers.append(nn.BatchNorm1d(latent_dim))
        if relu_out:
            layers.append(nn.ReLU())
        self.merge_1d = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
    def forward(self, x:dict):
        z_exp = self.module_1d(x['input'])
        z_img = self.module_2d(x['image'])
        z = self.merge_1d(torch.cat([z_exp, z_img], dim = 1))
        mu = self.fc_mu(z)
        std = torch.exp(0.5 * self.fc_var(z))
        eps = torch.randn_like(std)
        return eps * std + mu

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import math
import gc
import optuna
from optuna.pruners import MedianPruner

@dataclass
class hyper_parameters:
    # Architecture
    y_dim: int = 3         # Strain input dimension (Voigt: E11, E22, E12)
    u_dim: int = 32        # Latent geometry dimension (Z from Encoder)
    z_dim: int = 512       # Hidden dimension for cICNN physics layers
    num_layers: int = 4    # Number of hidden cICNN layers
    
    # Training Loop
    epochs: int = 500
    batch_size: int = 256
    str_sample: int = 600
    lr: float = 1e-3
    weight_decay: float = 1e-4  # AdamW penalty (crucial for smooth manifolds)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    stress_weight: float = 1.0
    phys_weight: float = 1.0
    kl_weight: float = 0.1
    varW: float = 1.0
    varS: float = 1.0
    nats_per_dim: float = 2.5
    # Checkpointing
    save: bool = False
    save_every_n_epochs: int = 50
    checkpoint_dir: str = "./checkpoints"
    
    #optuna yes/no
    hp_sweep: bool = False
    
class cICNN_layer(nn.Module):
    def __init__(self, config: hyper_parameters):
        super().__init__()
        ''' strain stream use "e", latent stream use "z" or "m" '''
        # U-Path (Non-Convex Latent Space Input)
        self.W_uu = nn.Linear(config.u_dim, config.u_dim, bias=True)
        
        # Z-Path (Convex Strain Input)
        # latent space contribution
        self.W_zu = nn.Linear(config.u_dim, config.z_dim, bias=True)
        self.W_z  = nn.Linear(config.z_dim, config.z_dim, bias=False) 
        
        target_post_softplus = 1.0 / config.z_dim
        init_mean = math.log(math.exp(target_post_softplus) - 1.0) 
        
        nn.init.normal_(self.W_z.weight, mean=init_mean, std=0.01)
        
        # strain contribution
        self.W_u  = nn.Linear(config.u_dim, config.z_dim, bias=True) 
        
        # cross attention contribution
        self.W_yu = nn.Linear(config.u_dim, config.y_dim, bias=True)
        self.W_y  = nn.Linear(config.y_dim, config.z_dim, bias=False) 
        
    def forward(self, inputs):
        y, u_i, z_i = inputs
        
        # NON-CONVEX UPDATE: u_{i+1}
        u_next = F.silu(self.W_uu(u_i))
        
        # CONVEX UPDATE: z_{i+1}

        zu_p = z_i * F.softplus(self.W_zu(u_i)) 
       
        term1 = F.linear(zu_p, F.softplus(self.W_z.weight)) 

        input_gate = self.W_yu(u_i) 
        yu = y * input_gate 
        term2 = self.W_y(yu)
      
        term3 = self.W_u(u_i)
        
        terms = term1 + term2 + term3
        z_next = F.softplus(terms)**2 + 0.01*terms
        return (y, u_next, z_next)
    
class cICNN_NN(nn.Module):
    def __init__(self, config: hyper_parameters):
        super().__init__()
        
        # 1. Initial Processing of Latent Geometry (Z from your VAE)
        self.u_init = nn.Linear(config.u_dim, config.u_dim)
        
        # Layer 0 
        # z_1 = Softplus(W_y y + W_u u_0 + b)
        self.layer_0_y = nn.Linear(config.y_dim, config.z_dim, bias=False)
        self.layer_0_u = nn.Linear(config.u_dim, config.z_dim, bias=True)
        
        layers = []
        for _ in range(config.num_layers):
            layers.append(cICNN_layer(config))
        
        self.hidden_layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(config.z_dim, 1, bias=False)
        
        target_post_softplus = 1.0 / config.z_dim
        init_mean = math.log(math.exp(target_post_softplus) - 1.0) 
        
        nn.init.normal_(self.final_layer.weight, mean=init_mean, std=0.01)
        # nn.init.normal_(self.final_layer.weight, mean=0, std=0.01)
    def _forward_raw(self, y, u):
        while u.dim() < y.dim(): u = u.unsqueeze(1)
        u_0 = self.u_init(u)
        z_1 = F.softplus(self.layer_0_y(y) + self.layer_0_u(u_0))
        
        _, _, z_out = self.hidden_layers((y, u_0, z_1))
        
        W_unact = F.linear(z_out, F.softplus(self.final_layer.weight))
        W_pred = F.softplus(W_unact)**2 + 0.01*W_unact
        # W_pred
        return W_pred
    
    def forward(self, strain, geometry):
        """ This enforce 0 energy,stress at 0 strain """
        if not strain.requires_grad:
            strain.requires_grad_(True)
            
        W_e = self._forward_raw(strain, geometry)
        strain_zero = torch.zeros_like(strain, requires_grad=True)
        W_zero = self._forward_raw(strain_zero, geometry)
        
        S_zero = torch.autograd.grad(W_zero, strain_zero, torch.ones_like(W_zero), create_graph=True)[0]
        linear_offset = torch.sum(S_zero * strain, dim=-1, keepdim=True)
        W_phys = W_e - W_zero - linear_offset
        
        return W_phys

    def _forward(self, strain, geometry):
        
        W_pred = self._forward_raw(strain, geometry)
        
        return W_pred
    
class VAE_encoder(nn.Module):
    def __init__(self, config: hyper_parameters):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, padding_mode='circular'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, padding_mode='circular'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, padding_mode='circular'),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, padding_mode='circular'),
            nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        self.z_mu     = nn.Linear(4096, config.u_dim)
        self.z_logvar = nn.Linear(4096, config.u_dim)

    def reparameterize(self, mu, logvar):
        """ sampling the VAE latent space """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        hidden = self.encoder(x)
        mu     = self.z_mu(hidden)
        logvar = self.z_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class VAE_decoder(nn.Module):
    def __init__(self, config: hyper_parameters):
        super().__init__()
        self.fc_decode = nn.Linear(config.u_dim, 4096)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, z):
        hidden = self.fc_decode(z)
        hidden = hidden.view(-1, 256, 4, 4)
        reconstruction = self.decoder(hidden)
        return reconstruction

class surrogateNN_VAE(nn.Module):
    def __init__(self, config: hyper_parameters):
        super().__init__()
        self.encoder = VAE_encoder(config)
        self.decoder = VAE_decoder(config)
        self.energy_predictor = cICNN_NN(config)

    def forward(self, geometry, strains):
        z, mu, logvar = self.encoder(geometry)
        reconstruction = self.decoder(z)
        W_phys = self.energy_predictor(strains, mu)
        
        return W_phys, reconstruction, mu, logvar
    
class surrogateNN(nn.Module):
    def __init__(self, config: hyper_parameters):
        super().__init__()
        self.encoder = VAE_encoder(config)
        self.energy_predictor = cICNN_NN(config)

    def forward(self, geometry, strains):
        z, mu, logvar = self.encoder(geometry)
        W_phys = self.energy_predictor(strains, z)
        
        return W_phys, None, mu, logvar    
class VRAMStorage:
    """Loads HDF5 data into VRAM, pre-computes physical fields, and excludes NaNs/Infs."""
    @torch.no_grad()
    def __init__(self, h5_path, device='cuda'):
        print(f"Bypassing I/O: Loading H5 into GPU VRAM, Pre-computing, and Filtering...")
        bad_indices = []
        
        with h5py.File(h5_path, 'r') as f:
            total_len = len(f['topologies'])
            print(f"Found {total_len} total samples. Processing in chunks...")
            
            topo_list = []
            energy_list = []
            strain_list = []
            stress_list = []
            
            chunk_size = 5000 
            
            # Setup constants for physics compute
            I = torch.eye(2, dtype=torch.float32, device=device).view(1, 1, 1, 2, 2)
            alphas = torch.linspace(0, 1.0, 11, device=device)[1:].view(1, 1, 10, 1, 1)
            
            for i in range(0, total_len, chunk_size):
                end = min(i + chunk_size, total_len)
                
                # 1. Load chunk to VRAM
                chunk_topo = torch.from_numpy(f['topologies'][i:end]).view(-1, 1, 64, 64).float().to(device)
                chunk_energy = torch.from_numpy(f['strain_energy'][i:end]).float().to(device) 
                chunk_U_max = torch.from_numpy(f['strain'][i:end]).float().to(device)
                chunk_P = torch.from_numpy(f['stress'][i:end]).float().to(device) 
                
                # 2. PRE-COMPUTE PHYSICS (Vectorized over chunk)
                # Expand dims if they are [batch, 300, 2, 2] to [batch, 300, 1, 2, 2]
                if chunk_U_max.dim() == 4: chunk_U_max = chunk_U_max.unsqueeze(2)
                if chunk_P.dim() == 4: chunk_P = chunk_P.unsqueeze(2)

                U = I + alphas * (chunk_U_max - I)
                E = 0.5 * (torch.matmul(U.mT, U) - I)
                
                # S = inv(U) @ P
                S = torch.matmul(torch.linalg.inv(U), chunk_P)
                
                # Convert to Voigt Notation: [batch, 300, 10, 3]
                chunk_strain = torch.stack([E[..., 0, 0], E[..., 1, 1], E[..., 0, 1]], dim=-1)
                chunk_stress = torch.stack([S[..., 0, 0], S[..., 1, 1], S[..., 0, 1]], dim=-1)
                
                # 3. Identify corrupted entries (now includes check against broken inverses!)
                # Flatten the last dims to safely catch NaNs across any varying shapes
                nan_mask = torch.isnan(chunk_energy).view(chunk_energy.shape[0], -1).any(dim=1) | \
                           torch.isnan(chunk_stress).view(chunk_stress.shape[0], -1).any(dim=1)
                           
                inf_mask = torch.isinf(chunk_energy).view(chunk_energy.shape[0], -1).any(dim=1) | \
                           torch.isinf(chunk_stress).view(chunk_stress.shape[0], -1).any(dim=1) 
                unphysical_mask = (torch.abs(chunk_energy) > 0.034).view(chunk_energy.shape[0], -1).any(dim=1)
                
                # Combine all failure modes
                invalid_mask = nan_mask | inf_mask | unphysical_mask
                valid_mask = ~invalid_mask  
                
                num_invalid = invalid_mask.sum().item()
                
                # 4. Filter and Append
                if num_invalid > 0:
                    corrupted_in_chunk = invalid_mask.nonzero(as_tuple=True)[0]
                    bad_indices.extend((corrupted_in_chunk + i).tolist())
                    
                    # Log the specific cause for debugging purposes
                    num_nan_inf = (nan_mask | inf_mask).sum().item()
                    num_unphys = unphysical_mask.sum().item()
                    print(f"    Chunk [{i}:{end}] -> Dropping {num_invalid} materials "
                          f"({num_unphys} unphysical limit | {num_nan_inf} NaN/Inf)")
                    
                    chunk_topo = chunk_topo[valid_mask]
                    chunk_energy = chunk_energy[valid_mask]
                    chunk_strain = chunk_strain[valid_mask]
                    chunk_stress = chunk_stress[valid_mask]
                
                topo_list.append(chunk_topo)
                energy_list.append(chunk_energy)
                strain_list.append(chunk_strain)
                stress_list.append(chunk_stress)
                
                if (i // chunk_size) % 2 == 0: 
                    print(f"Progress: {end}/{total_len} processed.")
        
        print(f"Concatenating filtered chunks into contiguous VRAM blocks...")
        
        # 5. Assemble the final clean, PRE-COMPUTED dataset
        self.topologies = torch.cat(topo_list, dim=0)
        self.energies = torch.cat(energy_list, dim=0)
        self.strains = torch.cat(strain_list, dim=0)     # Now fully computed Voigt E
        self.stresses = torch.cat(stress_list, dim=0)    # Now fully computed Voigt S
        
        self.length = len(self.topologies)
        self.corrupted_indices = set(bad_indices)
        
        if len(bad_indices) > 0:
            print(f"\n[!] WARNING: Excluded {len(bad_indices)} corrupted samples with NaN/Inf values.")
            print(f"[!] Original size: {total_len} -> Clean size: {self.length}")
        else:
            print("\n[✓] Data audit complete: No NaNs or Infs detected.\n")
            
        print("\nApplying Origin-Preserving Global Scaling...")
        
        #alternate scaling
        self.std_E = torch.max(torch.abs(self.strains))
        self.std_W = torch.max(torch.abs(self.energies))

        # self.std_E = torch.std(self.strains)
        # self.std_W = torch.std(self.energies)
        
        self.strains = self.strains / self.std_E
        self.energies = self.energies / self.std_W
        
        # Because S = dW/dE, the stress scaling should naturally be: S * (std_E / std_W)
        self.scale_S_multiplier = self.std_E / self.std_W
        self.stresses = self.stresses * self.scale_S_multiplier
        
        print(f"[✓] Scaling Factors -> Strain div: {self.std_E:.4f} | Energy div: {self.std_W:.4f} | Stress mult: {self.scale_S_multiplier:.4f}\n")

    def get_unscale_factors(self):
        """Returns the scalars needed to convert NN predictions back to real physics during deployment."""
        return {
            "strain_std": self.std_E.item(),
            "energy_std": self.std_W.item(),
            "stress_multiplier": self.scale_S_multiplier.item()
        }
    

class MetaMaterialDatasetPL(Dataset):
    """A 'View' of the VRAMStorage that decides whether to augment or not."""
    def __init__(self, storage, augment=False, device='cuda'):
        super().__init__()
        self.storage = storage
        self.augment = augment
        self.device = device
        
        # Pre-allocate the rotation multiplier for Voigt [11, 22, 12] -> [22, 11, -12]
        self.voigt_rot_mult = torch.tensor([1.0, 1.0, -1.0], device=self.device)
        
    def __len__(self): 
        return self.storage.length
        
    def __getitem__(self, idx):
        scaling = 1.0
        image = self.storage.topologies[idx]
        energy = self.storage.energies[idx]
        strain = self.storage.strains[idx]    # Pre-computed E [300, 10, 3]
        stress = self.storage.stresses[idx]   # Pre-computed S [300, 10, 3]
        
        if self.augment:
            # 90-degree spatial image rotation
            image = torch.rot90(image, 1, dims=[1, 2])
            
            # Mathematical 90-degree rotation in Voigt Notation: 
            # Swap 11 and 22, and multiply 12 by -1
            strain = strain[..., [1, 0, 2]] * self.voigt_rot_mult
            stress = stress[..., [1, 0, 2]] * self.voigt_rot_mult
            
        return image, strain, energy/scaling, stress/scaling
    
class FlattenedStrainDataset(Dataset):
    def __init__(self, material_dataset, num_strains=300, num_time=10):
        self.dataset     = material_dataset
        self.num_strains = num_strains
        self.num_time    = num_time
        self.points_per_mat = num_strains * num_time 

    def __len__(self):
        return len(self.dataset) * self.points_per_mat

    def __getitem__(self, idx):
        # 1. Math to find 3D coordinates (Material, Strain Path, Time Step)
        mat_idx = idx // self.points_per_mat           # Which material?
        remainder = idx % self.points_per_mat          # Which point within that material?
        
        strain_idx = remainder // self.num_time        # Which strain path?
        time_idx = remainder % self.num_time           # Which time step along that path?

        # 2. Get the full sequence for that specific material
        image, strains, energies, stresses = self.dataset[mat_idx]

        # 3. Slice out the exactly 1 physical state at that specific time step
        # Since energies is [300, 10, 1], slicing [strain_idx, time_idx] returns shape [1]
        # Since stresses is [300, 10, 3], slicing [strain_idx, time_idx] returns shape [3]
        single_strain = strains[strain_idx, time_idx]
        single_energy = energies[strain_idx, time_idx]
        single_stress = stresses[strain_idx, time_idx]

        return image, single_strain, single_energy, single_stress
    
def compute_loss(W_pred, W_true, S_pred, S_true, recon_images, real_images, mu, logvar, kl_targ=80,stress_weight=0.01, phys_weight=1.0,kl_weight=0.0, varW=1.0, varS=1.0):
    
    # Physics Loss: Mean Squared Error of the Strain Energy Density
    # loss_energy = F.mse_loss(torch.log(1e-6+W_pred), torch.log(1e-6+W_true.unsqueeze(-1)))#/varW
    loss_energy = F.mse_loss(W_pred, W_true.unsqueeze(-1))/varW
    loss_stress = F.mse_loss(S_pred, S_true)/varS

    # loss_energy = F.mse_loss((W_pred/W_true.unsqueeze(-1)),torch.ones_like(W_pred))
    # loss_stress = F.mse_loss((S_pred+1e-3)/(S_true+1e-3), torch.ones_like(S_pred))

    loss_phys = loss_energy + stress_weight * loss_stress
    
    # VAE KL Divergence
    loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss_kl = loss_kl / real_images.size(0) # Normalize by batch size
    
    total_loss =  phys_weight*loss_phys + kl_weight*torch.abs(loss_kl - kl_targ)
    loss_AE =  loss_kl
    return total_loss, loss_AE, loss_energy, loss_stress

def save_model_checkpoint(epoch, model, optimizer, config, loss_val):
    """Saves the network state safely to disk."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(config.checkpoint_dir, f"surrogate_epoch_{epoch}.pth")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_val,
        # Save the config so you can easily rebuild the exact architecture during inference
        'config': config 
    }, checkpoint_path)
    
    print(f"--> [Checkpoint Saved] Epoch {epoch} at {checkpoint_path}")
def get_weights_epoch(epoch,config):
    Tmin = 20
    Tmax = min(100+Tmin,config.epochs//2)
    
    t = torch.clamp((torch.tensor(epoch)  - Tmin)/(Tmax - Tmin), min=0, max=1)
    str_max  = config.stress_weight 
    kl_max   = config.kl_weight 
    st_wt    = (str_max/2)*(1-torch.cos(t*torch.pi))
    kl_wt    =  (kl_max/2)*(1-torch.cos(t*torch.pi))
    return st_wt.item(), kl_wt.item()
    
def train_model(model, train_dataloader, val_dataloader, config, trial=None):
    
    varW, varS = config.varW, config.varS
    str_sample = config.str_sample
    frozen=None
    epochs,lr,wd = config.epochs, config.lr, config.weight_decay
    device = config.device
    st_wt, p_wt, kl_wt = config.stress_weight, config.phys_weight, config.kl_weight
    kl_t = config.u_dim*config.nats_per_dim
    for param in model.parameters():
        param.requires_grad = True
    if frozen is not None:
        print(f"--- Enforcing freeze strategy. Freezing components: {frozen} ---")
        for part in frozen:
            part_upper = part.upper()
            
            if part_upper == 'P':
                if hasattr(model, 'energy_predictor'):
                    for param in model.energy_predictor.parameters():
                        param.requires_grad = False
                    print("-> Physics/PICNN network layers FROZEN.")

            elif part_upper == 'E':
                if hasattr(model, 'encoder'):
                    for param in model.encoder.parameters():
                        param.requires_grad = False
                    print("-> VAE Encoder network layers FROZEN.")
            
            elif part_upper == 'D':
                if hasattr(model, 'decoder'):
                    for param in model.decoder.parameters():
                        param.requires_grad = False
                    print("-> VAE Decoder network layers FROZEN.")

    # Pass ONLY parameters that require gradients to the optimizer
    active_parameters = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = AdamW(active_parameters, lr=lr, weight_decay=wd)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=20)
    model.to(device)
    # apply_convexity_constraints(model)
    
    import time
    from collections import deque
    best_val_loss = float('inf')
    
    # Store the last N validation losses
    history_window = 10 
    recent_val_losses = deque(maxlen=history_window)
    for epoch in range(epochs):
        # ==========================
        #       TRAINING PHASE
        # ==========================
        model.train()
        # model.decoder.eval()
        epoch_loss = 0.0
        epoch_energy = 0.0
        epoch_stress = 0.0
        start_time = time.time()
        st_wt, kl_wt = get_weights_epoch(epoch,config)
        for batch_idx, (images, strains_all, energies_true, stresses_true) in enumerate(train_dataloader):
            images = images.to(device, dtype=torch.float32)
            strains_all = strains_all.to(device, dtype=torch.float32)
            energies_true = energies_true.to(device, dtype=torch.float32)
            stresses_true = stresses_true.to(device, dtype=torch.float32)
            
            if str_sample is None: 
                strains       =   strains_all.flatten(1,2)
                energies_true = energies_true.flatten(1,2)
                stresses_true = stresses_true.flatten(1,2)
            
            else:
                total_strain_points = strains_all.shape[1]*strains_all.shape[2]
                random_indices = torch.randint(0, total_strain_points, (str_sample,), device=device)

                strains       =   strains_all.flatten(1,2)[:,random_indices,...]
                energies_true = energies_true.flatten(1,2)[:,random_indices,...]
                stresses_true = stresses_true.flatten(1,2)[:,random_indices,...]
                # print(strains.shape)
            
            strains.requires_grad_(True)
            optimizer.zero_grad()
            
            # Forward pass
            W_pred, recon_images, mu, logvar = model(images, strains)
            
            # Stresses using autograd
            S_pred = torch.autograd.grad(W_pred, strains, torch.ones_like(W_pred), create_graph=True)[0]
            
            loss, l_ae, l_energy, l_stress = compute_loss(
                W_pred, energies_true, S_pred, stresses_true, recon_images, images, mu, logvar,
                varW=varW,varS=varS,stress_weight=st_wt, phys_weight=p_wt, kl_weight=kl_wt, kl_targ=kl_t)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
            optimizer.step()
            
            
            epoch_loss += l_ae.detach().item()
            epoch_energy += l_energy.detach().item()
            epoch_stress += l_stress.detach().item()
        warmup_scheduler.step()    
        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_train_energy = epoch_energy / len(train_dataloader)
        avg_train_stress = epoch_stress / len(train_dataloader)

        # ==========================
        #      VALIDATION PHASE
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_energy = 0.0
        val_epoch_stress = 0.0
        
        # Note: No `with torch.no_grad():` here because we need autograd for Stress!
        for batch_idx_val, (val_images, val_strains, val_energies, val_stresses) in enumerate(val_dataloader):
            val_images = val_images.to(device, dtype=torch.float32)
            val_strains = val_strains.to(device, dtype=torch.float32)
            val_energies = val_energies.to(device, dtype=torch.float32)
            val_stresses = val_stresses.to(device, dtype=torch.float32)
            
            val_strains = val_strains
            if str_sample is None: 
                val_strains  =  val_strains.flatten(1,2)
                val_energies = val_energies.flatten(1,2)
                val_stresses = val_stresses.flatten(1,2)
            
            else:
                total_strain_points = val_strains.shape[1]*val_strains.shape[2]
                random_indices = torch.randint(0, total_strain_points, (str_sample,), device=device)

                val_strains  =  val_strains.flatten(1,2)[:,random_indices,...]
                val_energies = val_energies.flatten(1,2)[:,random_indices,...]
                val_stresses = val_stresses.flatten(1,2)[:,random_indices,...]
            

            val_strains.requires_grad_(True)
            
            # Forward pass
            W_pred_val, recon_val, mu_val, logvar_val = model(val_images, val_strains)
            
            # Stresses using autograd
            S_pred_val = torch.autograd.grad(W_pred_val, val_strains, torch.ones_like(W_pred_val), create_graph=True)[0]
            
            val_loss, vl_ae, l_energy_val, l_stress_val = compute_loss(
                W_pred_val, val_energies, S_pred_val, val_stresses, recon_val, val_images, mu_val, logvar_val,
                varW=varW,varS=varS,stress_weight=st_wt, phys_weight=p_wt, kl_targ=kl_t)
            
            val_epoch_loss += l_ae.detach().item()
            val_epoch_energy += l_energy_val.detach().item()
            val_epoch_stress += l_stress_val.detach().item()
            
        avg_val_loss = val_epoch_loss / len(val_dataloader)
        avg_val_energy = val_epoch_energy / len(val_dataloader)
        avg_val_stress = val_epoch_stress / len(val_dataloader)
        
        # ==========================
        #         PRINTING
        # ==========================
        print(f"Epoch [{epoch+1}/{epochs}] Time: {time.time() - start_time:.1f}s")
        print(f"  [Train] AE Loss: {avg_train_loss:.4f} | Energy MSE: {avg_train_energy:.6f} | Stress MSE: {avg_train_stress:.6f}")
        print(f"  [Val]   AE Loss: {avg_val_loss:.4f} | Energy MSE: {avg_val_energy:.6f} | Stress MSE: {avg_val_stress:.6f}\n")
        if config.save==True and epoch%50==0:
            save_model_checkpoint(epoch, model, optimizer, config, avg_train_loss)
        
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if math.isnan(avg_val_loss) or math.isinf(avg_val_loss):
                raise optuna.exceptions.TrialPruned()
        
        diff = np.abs(avg_val_loss/kl_t - 1)
        if diff > 0.1:
            out_loss = 100 + avg_val_energy + st_wt*avg_val_stress
        else:
            out_loss = avg_val_energy + st_wt*avg_val_stress
        recent_val_losses.append(out_loss)
    
    best_val_loss = min(recent_val_losses) if recent_val_losses else float('inf')     
    if config.save==True: save_model_checkpoint(epoch, model, optimizer, config, avg_train_loss)
    return best_val_loss
              
def optuna_objective(trial, train_dataset, val_dataset):

    config = hyper_parameters(  epochs        =  200,
                                u_dim         =  trial.suggest_categorical("u_dim", [8, 16, 32, 64]),
                                z_dim         =  trial.suggest_categorical("z_dim", [64, 128, 266, 512]),
                                num_layers    =  trial.suggest_int("num_layers", 2, 6),
                                lr            =  trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                                weight_decay  =  trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
                                batch_size    =  trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256]),
                                str_sample    =  trial.suggest_categorical("str_sample", [32, 64, 128, 256, 512, 1024]),
                                stress_weight =  trial.suggest_float("stress_weight", 1e-3, 10, log=True),
                                kl_weight     =  trial.suggest_float("kl_weight", 1e-6, 1.0, log=True)
                              )
    dataloader_train = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    dataloader_val   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False,num_workers=0, pin_memory=False)

    model = surrogateNN(config)
    loss  = train_model(model, dataloader_train, dataloader_val, config, trial=trial)
    
    return loss

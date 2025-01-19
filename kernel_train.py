import h5py
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed: int):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_value = 42
set_seed(seed_value)


gpu_idx = 7
device = torch.device(f"cuda:{gpu_idx}")
print(device)

n_exps = 1000000
neg_rng, pos_rng = -0.3, 0.3
t_min, t_max = -0.6, 0.6
neg_obs, pos_obs = -0.9, 0.9
a_min, a_max = 0.5, 10
n_samples = 11
snr_dB = 15
t_rng = torch.linspace(neg_obs, pos_obs, n_samples).to(device)
max_lr = 6e-4
min_lr = max_lr * 0.1
n_epochs = 1000
scale = 5

def add_noise(signals, snr_dB):

    signal_power = torch.mean(signals ** 2, dim=1, keepdim=True)
    snr_linear = 10 ** (snr_dB / 10)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power) * torch.randn_like(signals)
    return signals + noise


def refresh_data(data):
    data_len = data["amps"].shape[0]
    new_idx = np.random.permutation(data_len)
    data["amps"] = data["amps"][new_idx]
    data["locs"] = data["locs"][new_idx]
    return data

def mse_db(pred, actual):
    mse = np.mean((pred - actual)**2, axis=1)
    signal_power = np.mean(actual**2, axis=1)
    
    x = 10 * np.log10(mse / signal_power)

    return x

def get_batch(data, model, batch_size, idx):

    amp_batch = data["amps"][idx * batch_size : (idx + 1) * batch_size]
    loc_batch = data["locs"][idx * batch_size : (idx + 1) * batch_size]

    return model.get_signal(amp_batch, loc_batch, t_rng), loc_batch


def train_model(model, data, batch_size, n_epochs, optimizer, lr_scheduler):

    model.train()

    for epoch in range(n_epochs):
        data = refresh_data(data)
        n_batches = data["amps"].shape[0] // batch_size
        epoch_loss = 0

        for i in range(n_batches):
            optimizer.zero_grad()
            sigs, locs = get_batch(data, model, batch_size, i)
            sigs = add_noise(sigs, snr_dB)
            preds = model(sigs)
            loss = F.l1_loss(preds, locs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} / {n_epochs}, Loss: {epoch_loss / n_batches}")
        lr_scheduler.step()
    model.eval()

    return model

train_file = "train_40dB_3loc.h5"
test_file = "test_40dB_3loc.h5"
# train_file = "train_40dB.h5"
# test_file = "test_40dB.h5"


with h5py.File(train_file, 'r') as f:

    amp2 = f['/a_k'][:]
    loc2 = f['/t_k'][:]

aks = []
tks = []

for i in range(n_exps):
    a_values = amp2[i]
    t_values = loc2[i]

    tks.append(t_values)
    aks.append(a_values)

tks = np.array(tks)
aks = np.array(aks)
tks = torch.tensor(tks).float().to(device)
aks = torch.tensor(aks).float().to(device)

data = {"amps": aks, "locs": tks}

class FRIModel(nn.Module):
    def __init__(self, n_inp, n_out, neg_rng, pos_rng, scale):
        super(FRIModel, self).__init__()

        self.n_inp = n_inp
        self.n_brd = scale * n_inp
        self.neg_rng = neg_rng
        self.pos_rng = pos_rng

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * n_inp, 128)
        self.fc2 = nn.Linear(128, n_out)
        self.brd_pts = torch.linspace(neg_rng, pos_rng, scale * n_inp).to(device)

        ## Gaussian
        # self.coeffs = torch.exp(-(self.brd_pts**2)/(2 * (0.09 ** 2)))

        ## Gaussian pair
        # self.coeffs = 1.4*(torch.exp(-(self.brd_pts + 0.2)**2 / (2 * (0.045 ** 2))) + torch.exp(-(self.brd_pts - 0.19)**2 / (2 * (0.045 ** 2))))
        # self.coeffs = 1.25 * torch.exp(-(self.brd_pts + 0.1)**2 / (2 * (0.06 ** 2))) + 0.7 * torch.exp(-(self.brd_pts - 0.16)**2 / (2 * (0.07 ** 2)))

        ## Initialize as gaussian
        # self.coeffs = nn.Parameter(torch.exp(-(self.brd_pts**2)/(2 * (self.pos_rng ** 2))))
        self.coeffs = nn.Parameter(torch.exp(-(self.brd_pts**2)/(2 * (0.09 ** 2))))


    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_ker(self):
        return self.coeffs

    def get_brd_pts(self):
        return self.brd_pts


    def fn_val(self, c1, c2, strt, t_val, sep):

        ret_val = (c1 + ((c2 - c1) * (t_val - strt)) / sep).sum(dim=1)
        return ret_val


    def get_spikes(self, aks, tks):
        ker = self.get_ker().unsqueeze(0).unsqueeze(1)
        a = aks.unsqueeze(2) * ker
        t = tks.unsqueeze(2) + self.brd_pts

        return a, t

    def get_sig(self, t_samps, a, t):

        diffs = torch.abs(t.unsqueeze(-1) - t_samps.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(t.shape[0], t.shape[1], t.shape[2], -1))
        min_ids = torch.argmin(diffs, dim=-2)

        t_fs = torch.gather(t, 2, min_ids)
        t_samps_exp = t_samps.unsqueeze(0).unsqueeze(0).expand(t_fs.shape[0], t_fs.shape[1], -1)

        cond_1 = (t_fs < t_samps_exp) & (diffs[:, :, self.n_brd // 2, :] < self.pos_rng)
        cond_2 = (t_fs >= t_samps_exp) & (diffs[:, :, self.n_brd // 2, :] < self.pos_rng)
        cond_3 = (min_ids == 0) & (diffs[:,:,self.n_brd // 2, :] > self.pos_rng)
        cond_4 = (min_ids == self.n_brd - 1) & (diffs[:,:,self.n_brd // 2, :] > self.pos_rng)
        cond_5 = (min_ids == 0) & (diffs[:,:,self.n_brd // 2, :] < self.pos_rng)
        cond_6 = (min_ids == self.n_brd - 1) & (diffs[:,:,self.n_brd // 2, :] < self.pos_rng)

        min_ids[cond_3] = 1
        min_ids[cond_5] = 1
        min_ids[cond_4] = self.n_brd - 2
        min_ids[cond_6] = self.n_brd - 2

        strt_pts = torch.where(cond_3 | cond_4, -100, torch.where(cond_1, torch.where(cond_5, torch.gather(t, 2, min_ids - 1), torch.gather(t, 2, min_ids)), torch.where(cond_6, torch.gather(t, 2, min_ids), torch.gather(t, 2, min_ids - 1))))
        c1_vals = torch.where(cond_3 | cond_4, 0, torch.where(cond_1, torch.where(cond_5, torch.gather(a, 2, min_ids - 1), torch.gather(a, 2, min_ids)), torch.where(cond_6, torch.gather(a, 2, min_ids), torch.gather(a, 2, min_ids - 1))))
        c2_vals = torch.where(cond_3 | cond_4, 0, torch.where(cond_2, torch.where(cond_6, torch.gather(a, 2, min_ids + 1), torch.gather(a, 2, min_ids)), torch.where(cond_5, torch.gather(a, 2, min_ids), torch.gather(a, 2, min_ids + 1))))

        return self.fn_val(c1_vals, c2_vals, strt_pts, t_samps, self.brd_pts[1] - self.brd_pts[0])

    def get_signal(self, aks, tks, t_rng):
        a_, t_ = self.get_spikes(aks, tks)
        sig = self.get_sig(t_rng, a_, t_)
        return sig
    
    
model = FRIModel(n_samples, 3, neg_rng, pos_rng, scale).to(device)

optimizer = optim.AdamW(model.parameters(), lr=max_lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=min_lr)
        
n_params = 0

for n, p in model.named_parameters():
    if(p.requires_grad):
        n_params += p.numel()
print('Number of parameters: ', n_params)

model = train_model(model, data, 8192, n_epochs, optimizer, scheduler)
torch.save(model.state_dict(), f"l1_loss/kernel_locs3_samples11_15dB.pt")

for param in model.parameters():
    param.requires_grad = False

def mse_fit(coeffs, dataloader, loss_fn, epochs=30):

    coeffs.requires_grad_(True)

    for i in range(epochs):
        total_loss = 0
        for idx, batch in enumerate(dataloader):
            locs, sig_true = batch
            locs, sig_true = locs.to(device), sig_true.to(device)

            start_idx = idx*locs.shape[0]
            end_idx = min((idx + 1)*locs.shape[0], coeffs.shape[0])

            amp_batch = coeffs[start_idx:end_idx, :]

            sig_pred = model.get_signal(amp_batch, locs, t_rng)

            loss = loss_fn(sig_pred, sig_true)
            optimizer.zero_grad()

            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {i+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

    coeffs.requires_grad_(False)

    return coeffs

with h5py.File(test_file, 'r') as f:

    amp3_test = f['/a_k'][:]
    loc3_test = f['/t_k'][:]

n_exps = 1000

tks_test = []
aks_test = []

for i in range(n_exps):

    a_values = amp3_test[i]
    t_values = loc3_test[i]

    tks_test.append(t_values)
    aks_test.append(a_values)

tks_test = torch.tensor(np.array(tks_test)).float().to(device)
aks_test = torch.tensor(np.array(aks_test)).float().to(device)

data_test = {"amps": aks_test, "locs": tks_test}

sig_test, loc_test = get_batch(data_test, model, len(aks_test), 0)
sig_test = add_noise(sig_test, snr_dB)
model.eval()

tks_pred = model(sig_test).cpu().detach().numpy()

tks_pred = torch.tensor(np.array(tks_pred)).float().to(device)

aks_pred = (10 * torch.randn_like(aks_test) + 10) / 2
optimizer = optim.Adam([aks_pred], lr=0.05)
train = TensorDataset(tks_pred, sig_test)
train = DataLoader(train, batch_size=64, shuffle=False)

aks_test = aks_test.cpu().detach().numpy()
tks_pred = tks_pred.cpu().detach().numpy()
tks_test = tks_test.cpu().detach().numpy()

aks_pred = mse_fit(aks_pred, train, F.mse_loss, epochs=1)
aks_pred = aks_pred.cpu().detach().numpy()
del train, sig_test

tks_mse, aks_mse = mse_db(tks_pred, tks_test), mse_db(aks_pred, aks_test)

print("Location prediction MSE (dB) :", np.mean(tks_mse))
print("Amplitude prediction MSE (dB) :", np.mean(aks_mse))
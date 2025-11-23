import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_ddpm(
        model: torch.nn.Module,
        train_loader: DataLoader,
        alphas_bar: torch.Tensor,
        T: int,
        device: torch.device,
        epochs: int = 60,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> torch.nn.Module:
    """
    Train a Denoising Diffusion Probabilistic Model (DDPM).
    Args:
        model (torch.nn.Module): The DDPM model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        alphas_bar (torch.Tensor): Cumulative product of alphas for the diffusion process.
        T (int): Number of timesteps in the diffusion process.
        device (torch.device): Device to run the training on.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        weight_decay (float): Weight decay for the optimizer.
    Returns:
        torch.nn.Module: The trained DDPM model.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for (batch,) in pbar:
            batch = batch.to(device)
            batch_size = batch.size(0)

            t = torch.randint(0, T, (batch_size,), device=device).long()
            
            alphas_bar_t = alphas_bar[t].view(-1, 1)
            one_minus_alphas_bar_t = 1 - alphas_bar_t

            noise = torch.randn_like(batch)
            x_t = torch.sqrt(alphas_bar_t) * batch + torch.sqrt(one_minus_alphas_bar_t) * noise

            t01 = ( t.float() + 0.5 ) / T
            noise_pred = model(x_t, t01)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

    return model

import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, device, epochs, custom_loss):
    """
    Trains the model and returns training losses and forecasted demand predictions.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the model on.
        epochs (int): Number of epochs to train.
        custom_loss (function): Custom loss function.

    Returns:
        train_losses (list): List of training losses per epoch.
        df_pred (torch.Tensor): Forecasted demand predictions.
    """
    train_losses = []
    df_pred = None  # To store forecasted demand predictions

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
            seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
            df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

            # Forward pass
            df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)
            loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
                seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
                df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

                df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)
                loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

    return train_losses, df_pred
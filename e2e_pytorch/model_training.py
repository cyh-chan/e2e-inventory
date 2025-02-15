import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, device, epochs, custom_loss):
    train_losses = []
    val_losses = []
    train_mse = []
    val_mse = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse_sum = 0.0
        train_total = 0

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

            # Calculate MSE for df_pred and df_true
            mse = torch.mean((df_pred - df_true.unsqueeze(1)) ** 2)  # Mean Squared Error
            train_mse_sum += mse.item()
            train_total += 1

        # Average train loss and MSE for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mse = train_mse_sum / train_total  # Average MSE over batches
        train_losses.append(avg_train_loss)
        train_mse.append(avg_train_mse)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train MSE: {avg_train_mse:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse_sum = 0.0  # Sum of MSE for all validation batches
        val_total = 0      # Total number of validation batches
        with torch.no_grad():
            for batch in val_loader:
                seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
                seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
                df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

                df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)
                loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))
                val_loss += loss.item()

                # Calculate MSE for validation
                mse = torch.mean((df_pred - df_true.unsqueeze(1)) ** 2)
                val_mse_sum += mse.item()
                val_total += 1

        # Average validation loss and MSE for the epoch
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse_sum / val_total  # Average MSE over validation batches
        val_losses.append(avg_val_loss)
        val_mse.append(avg_val_mse)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation MSE: {avg_val_mse:.4f}")

    return train_losses, val_losses, train_mse, val_mse, df_pred
import torch
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, device, epochs, custom_loss):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
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

            # Calculate accuracy (assuming regression)
            # For regression tasks, use Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE)
            # Here, we calculate MAE for df_pred and df_true
            mae = torch.mean(torch.abs(df_pred - df_true.unsqueeze(1)))  # Align shapes
            train_correct += mae.item()  # Use MAE as a proxy for accuracy
            train_total += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / train_total  # Average MAE over batches
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
                seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
                df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

                df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)
                loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))
                val_loss += loss.item()

                # Calculate accuracy (MAE for validation)
                mae = torch.mean(torch.abs(df_pred - df_true.unsqueeze(1)))
                val_correct += mae.item()
                val_total += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_correct / val_total  # Average MAE over batches
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}, Validation MAE: {avg_val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies, df_pred
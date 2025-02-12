import torch

def evaluate_model(model, val_loader, device, custom_loss):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in, df_true, rp_true, vlt_true = batch
            seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in = seq_in.to(device), cat_fea_in.to(device), vlt_in.to(device), rp_in.to(device), initial_stock_in.to(device)
            df_true, rp_true, vlt_true = df_true.to(device), rp_true.to(device), vlt_true.to(device)

            df_pred, layer4_pred, vlt_pred = model(seq_in, cat_fea_in, vlt_in, rp_in, initial_stock_in)
            loss = custom_loss((df_true, rp_true, vlt_true), (df_pred, layer4_pred, vlt_pred))
            test_loss += loss.item()

    avg_test_loss = test_loss / len(val_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss
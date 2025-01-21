import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class E2EModel(nn.Module):
    def __init__(
            self,
            seq_len,
            n_dyn_fea=1,
            n_outputs=1,
            n_dilated_layers=3,
            kernel_size=3,
            n_filters=3,
            dropout_rate=0.1,
            max_cat_id=[100, 100],
    ):
        super(E2EModel, self).__init__()
        self.seq_len = seq_len
        self.n_dyn_fea = n_dyn_fea
        self.n_cat_fea = len(max_cat_id)
        self.kernel_size = kernel_size
        self.n_dilated_layers = n_dilated_layers

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(m + 1, math.ceil(math.log(m + 1)))
            for m in max_cat_id
        ])

        # Dilated convolution layers
        self.conv_layers = nn.ModuleList()

        # First layer
        self.conv_layers.append(
            nn.Conv1d(
                n_dyn_fea,
                n_filters,
                kernel_size,
                padding='same',
                dilation=1
            )
        )

        # Subsequent layers
        for i in range(1, n_dilated_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size,
                    padding='same',
                    dilation=2 ** i
                )
            )

        # Final convolutional layer
        self.final_conv = nn.Conv1d(n_filters * 2 if n_dilated_layers > 1 else n_filters, 8, 1)

        # Calculate the flattened size for conv output
        conv_out_size = 8 * seq_len

        # Embedding total size
        embed_total_size = sum(math.ceil(math.log(m + 1)) for m in max_cat_id)

        # Dense layers
        self.dense1 = nn.Linear(conv_out_size + embed_total_size, 64)
        self.output_layer = nn.Linear(64, n_outputs)

        # VLT layers
        self.vlt_dense1 = nn.Linear(seq_len + embed_total_size, 16)
        self.vlt_dense2 = nn.Linear(16, 1)

        # Layer 3 (Reorder point)
        self.layer3_dense = nn.Linear(64 + 16 + seq_len, 32)

        # Layer 4 (Initial stock)
        self.layer4_dense1 = nn.Linear(32 + seq_len, 64)
        self.layer4_dense2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, seq_in, cat_fea_in, vlt_input, rp_in, initial_stock_in):
        # Process sequential input
        x = seq_in.transpose(1, 2)  # Convert to (batch, channels, seq_len)

        # Apply dilated convolutions
        conv_outputs = []

        # First layer
        conv_out = self.conv_layers[0](x)
        conv_outputs.append(F.relu(conv_out))

        # Subsequent layers
        for i, conv in enumerate(self.conv_layers[1:], 1):
            conv_out = conv(conv_outputs[-1])
            conv_outputs.append(F.relu(conv_out))

        # Skip connection
        if len(self.conv_layers) > 1:
            conv_out = torch.cat([conv_outputs[0], conv_outputs[-1]], dim=1)
        else:
            conv_out = conv_outputs[0]

        # Final convolution
        conv_out = F.relu(self.final_conv(conv_out))
        conv_out = self.dropout(conv_out)
        conv_out = conv_out.flatten(1)

        # Process categorical features
        cat_outputs = []
        for i, embed in enumerate(self.embeddings):
            cat_out = embed(cat_fea_in[:, i])
            cat_outputs.append(cat_out.squeeze(1))

        # Concatenate conv output with categorical features
        x = torch.cat([conv_out] + cat_outputs, dim=1)
        x = F.relu(self.dense1(x))
        df_output = self.output_layer(x)  # shape: [batch_size, forecast_horizon]

        # VLT processing
        vlt_cat = torch.cat([vlt_input.flatten(1)] + cat_outputs, dim=1)
        vlt_hidden = F.relu(self.vlt_dense1(vlt_cat))
        vlt_hidden = self.dropout(vlt_hidden)
        vlt_output = F.relu(self.vlt_dense2(vlt_hidden))  # shape: [batch_size, 1]

        # Layer 3 (Reorder point)
        layer3_input = torch.cat([x, vlt_hidden, rp_in.flatten(1)], dim=1)
        layer3_output = F.relu(self.layer3_dense(layer3_input))
        layer3_output = self.dropout(layer3_output)

        # Layer 4 (Initial stock)
        layer4_input = torch.cat([layer3_output, initial_stock_in.flatten(1)], dim=1)
        layer4_hidden = F.relu(self.layer4_dense1(layer4_input))
        layer4_hidden = self.dropout(layer4_hidden)
        layer4_output = F.relu(self.layer4_dense2(layer4_hidden))  # shape: [batch_size, 1]

        return df_output, layer4_output, vlt_output


def custom_loss(y_true, y_pred):
    # Unpack predictions and targets
    df_true, layer4_true, vlt_true = y_true
    df_pred, layer4_pred, vlt_pred = y_pred

    # Ensure consistent dimensions
    layer4_true = layer4_true.view(-1, 1)  # reshape to [batch_size, 1]
    vlt_true = vlt_true.view(-1, 1)  # reshape to [batch_size, 1]

    # Calculate MSE for each output
    mse_df = F.mse_loss(df_pred, df_true)
    mse_layer4 = F.mse_loss(layer4_pred, layer4_true)
    mse_vlt = F.mse_loss(vlt_pred, vlt_true)

    # Weighted sum of losses
    loss = 0.7 * mse_df + 0.25 * mse_layer4 + 0.05 * mse_vlt

    return loss

# def custom_loss(y_true, y_pred):
#     # Unpack predictions and targets
#     df_true, layer4_true, vlt_true = y_true
#     df_pred, layer4_pred, vlt_pred = y_pred
#
#     # Ensure consistent dimensions
#     layer4_true = layer4_true.view(-1, 1)  # reshape to [batch_size, 1]
#     vlt_true = vlt_true.view(-1, 1)  # reshape to [batch_size, 1]
#
#     # Apply scaling if necessary
#     scale_df = 1.0 / (torch.std(df_true) + 1e-6)  # Avoid division by zero
#     scale_layer4 = 1.0 / (torch.std(layer4_true) + 1e-6)
#     scale_vlt = 1.0 / (torch.std(vlt_true) + 1e-6)
#
#     # Calculate scaled MSE for each output
#     mse_df = scale_df * F.mse_loss(df_pred, df_true)
#     mse_layer4 = scale_layer4 * F.mse_loss(layer4_pred, layer4_true)
#     mse_vlt = scale_vlt * F.mse_loss(vlt_pred, vlt_true)
#
#     # Weighted sum of losses with dynamic weighting
#     total_weight = mse_df + mse_layer4 + mse_vlt
#     w_df = 0.6 * (mse_df / total_weight).detach()
#     w_layer4 = 0.3 * (mse_layer4 / total_weight).detach()
#     w_vlt = 0.1 * (mse_vlt / total_weight).detach()
#
#     # Compute total loss
#     loss = w_df * mse_df + w_layer4 * mse_layer4 + w_vlt * mse_vlt
#
#     return loss

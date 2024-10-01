import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from processing_eeg_methods.data_utils import (
    get_input_data_path,
    standard_saving_path,
)
from processing_eeg_methods.share import datasets_basic_infos

from .diffE_models import Decoder, DiffE, Encoder, LinearClassifier
from .diffE_utils import EEGDataset

dataset_name = "aguilera_traditional"  # Only two things I should be able to change

data_path: str = get_input_data_path(dataset_name)
dataset_info = datasets_basic_infos[dataset_name]  # Changed to get the dataclass instance

def diffE_evaluation(subject_id: int, X, Y, dataset_info, device: str = "cuda:0"):
    X = X[:, :, : -1 * (X.shape[2] % 8)]  # 2^3=8 because there are 3 downs and ups halves.
    
    # Dataloader
    batch_size2 = 260
    train_loader = DataLoader(EEGDataset(X, Y), batch_size=batch_size2, shuffle=False)

    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512
    
    # Define model
    model_path: str = standard_saving_path(
        dataset_info.dataset_name, "DiffE", "", file_ending="pt", subject_id=subject_id
    )
    num_classes = dataset_info.num_classes  # Changed to access the dataclass attribute
    channels = dataset_info.num_channels  # Changed to access the dataclass attribute

    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    decoder = Decoder(
        in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
    ).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    # Load the pre-trained model from the file
    diffe.load_state_dict(torch.load(model_path))

    diffe.eval()

    with torch.no_grad():
        Y = []
        Y_hat = []
        for x, y in train_loader:
            x, y = x.to(device).float(), y.type(torch.LongTensor).to(device)
            encoder_out = diffe.encoder(x)
            y_hat = diffe.fc(encoder_out[1])
            y_hat = F.softmax(y_hat, dim=1)

            Y.append(y.detach().cpu())
            Y_hat.append(y_hat.detach().cpu())

        # Convert list of tensors to numpy
        Y = torch.cat(Y, dim=0).numpy()  # (N, )
        Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

        return Y, Y_hat

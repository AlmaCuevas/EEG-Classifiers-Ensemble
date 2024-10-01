import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from processing_eeg_methods.data_utils import standard_saving_path

from .diffE_models import Decoder, DiffE, Encoder, LinearClassifier
from .diffE_utils import EEGDataset

# todo: do the deap thing about the FFT: https://github.com/tongdaxu/EEG_Emotion_Classifier_DEAP/blob/master/Preprocess_Deap.ipynb


def diffE_test(subject_id: int, X, dataset_info: dict, device: str = "cuda:0"):
    # From diffe_evaluation
    model_path: str = standard_saving_path(
        dataset_info.dataset_name, "DiffE", "", file_ending="pt", subject_id=subject_id
    )

    X = X[
        :, :, : -1 * (X.shape[2] % 8)
    ]  # 2^3=8 because there are 3 downs and ups halves.
    # Dataloader
    batch_size2 = 260
    testing_set = EEGDataset(
        X, [0] * (X.shape[0])
    )  # Y=0 JUST TO NOT LEAVE IT EMPTY, HERE IT ISN'T USED
    testing_loader = DataLoader(testing_set, batch_size=batch_size2, shuffle=False)

    n_T = 1000
    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512
    # Define model
    num_classes = dataset_info.num_classes  # Changed to access the dataclass attribute
    channels = dataset_info.num_channels  # Changed to access the dataclass attribute

    encoder = Encoder(in_channels=channels, dim=encoder_dim).to(device)
    decoder = Decoder(
        in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
    ).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    # load the pre-trained model from the file
    diffe.load_state_dict(torch.load(model_path))

    diffe.eval()

    with torch.no_grad():
        Y_hat = []
        for x, _ in testing_loader:
            x = x.to(device).float()
            encoder_out = diffe.encoder(x)
            y_hat = diffe.fc(encoder_out[1])
            y_hat = F.softmax(y_hat, dim=1)

            Y_hat.append(y_hat.detach().cpu())
        Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row
    return Y_hat

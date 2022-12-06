from config import config_parser
import torch
from pathlib import Path
import wandb
from datasets.base import build_dataloader
from models import net
from utils.loss_functions import landmark_loss, eye_closure_loss
from utils.utils import squared_norm


def save_model(network, epoch, optimizer, loss, args, best_model=False):
    print('Saving model epoch: ', epoch)
    if not Path(args.checkpoints_dir).exists():
        Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    if best_model:
        save_dir = Path(args.checkpoints_dir, 'model_' + str(epoch) + '_best.pt')
    else:
        save_dir = Path(args.checkpoints_dir, 'model_' + str(epoch) + '.pt')
    torch.save(
        {'epoch': epoch, 'model_state_dict': network.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
         'loss': loss}, save_dir)


def print_progress(loss, best_loss=None, mode='train'):
    print(mode + ' loss: ', float(loss))
    if best_loss:
        print('best ' + mode + ' loss: ', float(best_loss))


if __name__ == '__main__':
    # load configs
    parser = config_parser()
    args = parser.parse_args()

    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    # prepare data
    train_loader = build_dataloader(args, mode='train')
    val_loader = build_dataloader(args, mode='val')

    # build the network
    network = net.Encoder(args, device).to(device)

    # Build an optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    best_val_loss = None

    # initialize wandb
    # wandb.init(project='project-name', config=args, notes="pose params only", tags=["pose"], mode="offline")
    # wandb.watch(network)

    # Start the training process
    for epoch in range(args.continue_from_epoch, args.num_epochs):
        print('Training epoch: ', epoch)
        network.train()
        for batch_idx, data in enumerate(train_loader):
            # Load data
            inputs = data["image"].to(device)
            labels = data["landmarks"].to(device)
            tf_params = data["scale_ratio"]

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            prediction, _ = network(inputs, tf_params)

            # Find the loss
            lmk_loss, weighted_lmk_loss = landmark_loss(labels, prediction)
            loss = weighted_lmk_loss + eye_closure_loss(labels, prediction) + args.reg_term * (
                        squared_norm(network.expression_params) + squared_norm(network.shape_params))

            # Calculate gradients
            loss.backward()

            # Updates ws and bs
            optimizer.step()

            if batch_idx % args.log_every == 0 and epoch > 0:
                print_progress(loss, mode='train')
                with torch.no_grad():
                    # track also non-weighted loss as a comparison metric
                    wandb.log({'train loss': float(loss), 'non-weighted lmk loss train': float(lmk_loss)})

        if epoch % args.save_every == 0 and epoch > 0:
            save_model(network, epoch, optimizer, loss, args, best_model=False)

        # Start the validation process
        if epoch % args.val_every == 0 and epoch > 0:
            print('Validating epoch: ', epoch)
            val_loss = 0
            val_lmk_loss = 0
            count = 0

            network.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(val_loader):
                    # load data
                    inputs = data["image"].to(device)
                    labels = data["landmarks"].to(device)
                    tf_params = data["scale_ratio"]
                    # forward pass
                    prediction, _ = network(inputs, tf_params)

                    lmk_loss, weighted_lmk_loss = landmark_loss(labels, prediction)
                    loss = weighted_lmk_loss + eye_closure_loss(labels, prediction) + args.reg_term * (
                                squared_norm(network.expression_params) + squared_norm(network.shape_params))

                    val_loss += loss.detach()
                    val_lmk_loss += lmk_loss.detach()
                    count += 1

                val_loss = val_loss / count
                val_lmk_loss = val_lmk_loss / count
                print_progress(val_loss, best_val_loss, mode='val')
                wandb.log({'val loss': float(val_loss), 'non-weighted lmk loss val': float(val_lmk_loss)})

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(network, epoch, optimizer, best_val_loss, args, best_model=True)

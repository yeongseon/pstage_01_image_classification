import argparse
import os
import tarfile
from importlib import import_module

import torch
from torch.utils.data import DataLoader


def load_model(saved_model, num_classes):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    num_gpus = torch.cuda.device_count()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    output_file = open(os.path.join(output_dir, 'output.csv'), 'w')

    dataset_cls = getattr(import_module("dataset"), args.dataset)
    dataset = dataset_cls(
        data_dir=data_dir,
        phase='test',
    )

    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=use_cuda,
        drop_last=True,
    )

    model = load_model(model_dir, dataset.num_classes).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        outs = model(data)
        preds = outs.argmax(dim=1)
        preds = preds.detach().cpu().numpy()

        for pred in preds:
            output_file.write(f'{pred}\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--dataset', type=str, default='MaskMultiClassDataset', help='dataset type (default: MaskMultiClassDataset)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', '/model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    inference(data_dir, model_dir, output_dir, args)
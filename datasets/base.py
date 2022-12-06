from torch.utils.data import DataLoader
from torchvision import transforms
from .custom_datasets import vgg_face2


def build_dataset(args, mode):
    if args.dataset_type == "vgg_face2":
        transform_image = transforms.Compose([transforms.ToTensor(), transforms.Resize([args.height, args.width]),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
        if mode == "train": csv_file = args.csv_file_train
        elif mode == "val": csv_file = args.csv_file_val
        else: csv_file = args.csv_file_test
        dataset = vgg_face2(csv_file=csv_file, root_dir=args.root_dir, mode=mode, transform=transform_image)
    else:
        raise NotImplementedError()
    return dataset


def build_dataloader(args, mode):
    shuffle = True if mode == 'train' else False
    dataset = build_dataset(args, mode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return dataloader

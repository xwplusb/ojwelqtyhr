from torchvision.transforms import transforms


trans_dict = {
    'MNIST_train_trans':transforms.ToTensor(),
    'GTRSB_Transform': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((96,96), antialias=True)
    ])
}

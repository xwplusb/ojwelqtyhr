from torchvision.utils import make_grid, save_image

def save_img(t, nrow, path):
    t = t.detach().cpu()
    t = make_grid(t, nrow)
    save_image(t, path)


# Inherit Discriminator from nn.Module.
class Discriminator(nn.Module):

    def __init__(self, img_dim):
        super().__init__()

        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128), # taking an 128 neurons.
            nn.LeakyReLU(0.1), # slope of 0.1 taking.
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self,x): # x is some kinds of input.
        return self.disc(x)

# Inherit Generator from nn.Module.
class Generator(nn.Module):

    def __init__(self, z_dim, img_dim): # z_dim is the dimension of the latent noise.
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim), # 28 * 28 * 1 ---> 784
            nn.Tanh(),
        )

    def forward(self,x):
        return self.gen(x)

# Hyerparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64 # Can try with 128, 256
img_dim = 28 * 28 * 1 # 784.
batch_size = 32
num_epochs = 20

# Initialization of Discriminator.
disc = Discriminator(img_dim).to(device)

# Initialization of Generator.
gen = Generator(z_dim, img_dim).to(device)

# setting of the noise.
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# transforms.Compose() --> is a function that allows you to chain together multiple image transformation operations. In this case, It takes a list of transformations, applies them sequentially to an Image.

# transforms.ToTensor() --> is a transformation that converts a PIL image or a Numpy ndarray into a PyTorch tensor. Specifically, it converts the image data from a range of [0, 255] to a range of [0.0, 1.0]. It also rearranges the dimension of the images from (H, W, C) to (C, W, H) which is format expected by the PyTorch Models. where, H -> Height, W -> Weight, and C -> Channels.

# transforms.Normalize((0.1307,), (0.3081,)) transforms.Normalize is used to normalize the image tensor with a specified mean and standard deviation. Normalization is a common preprocessing step that helps to stabilize and speed up the training process of neural networks by ensuring that the input features have a similar scale. # (0.1307,) is the mean used for normalization. # (0.3081,) is the standard deviation used for normalization.

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 10

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

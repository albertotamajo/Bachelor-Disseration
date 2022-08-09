# Own modules
#from chamferdist import ChamferDistance
from pytorch3d.loss import chamfer_distance

# Third-party modules
import torch.optim
import torch.distributions as D
import os
import torch.distributed as dist
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans


def save_checkpoint(epoch, model, d_optimizer, enc_optimizer, g_optimizer, d_scheduler, vae_g_scheduler, train_d_losses,
                    train_enc_losses, train_g_losses, test_d_losses, test_enc_losses, test_g_losses,
                    d_learning_rates, enc_learning_rates, g_learning_rates, gaussian_mixture, save_path):

    dict = {
        'epoch': epoch,
        'model_state_dict': model.get_submodule("module").state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'enc_optimizer_state_dict': enc_optimizer.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'train_d_losses': train_d_losses,
        'train_enc_losses': train_enc_losses,
        'train_g_losses': train_g_losses,
        'test_d_losses': test_d_losses,
        'test_enc_losses': test_enc_losses,
        'test_g_losses': test_g_losses,
        'd_learning_rates': d_learning_rates,
        'enc_learning_rates': enc_learning_rates,
        'g_learning_rates': g_learning_rates,
        'gaussian_mixture': gaussian_mixture
    }
    if d_scheduler is not None:
        dict['d_scheduler_state_dict'] = d_scheduler.state_dict()
    if vae_g_scheduler is not None:
        dict['vae_g_scheduler_state_dict'] = vae_g_scheduler.state_dict()

    torch.save(dict, save_path)


def load_checkpoint(model, d_optimizer, enc_optimizer, g_optimizer, d_scheduler, vae_g_scheduler, load_path):
    try:
        checkpoint = torch.load(load_path)
        print("Progress file in the folder")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dictionary read")
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        print("Discriminator optimizer state dictionary read")
        enc_optimizer.load_state_dict(checkpoint['enc_optimizer_state_dict'])
        print("Encoder optimizer state dictionary read")
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        print("Generator optimizer state dictionary read")
        if d_scheduler is not None:
            d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
            print("Discriminator scheduler state dictionary read")
        if vae_g_scheduler is not None:
            vae_g_scheduler.load_state_dict(checkpoint['vae_g_scheduler_state_dict'])
            print("VAE-Generator scheduler state dictionary read")
    except:
        print("Progress file not in the folder")
        return 0


def load_gaussian_mixture(load_path):
    return torch.load(load_path)['gaussian_mixture']


def cluster(feature_matrix_batch, n_clusters, init, n_init):
    feature_matrix_batch = torch.clone(feature_matrix_batch).detach().cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init)
    centroids_batch = []
    for feature_matrix in feature_matrix_batch:
        kmeans.fit(feature_matrix)
        centroids_batch.append(kmeans.cluster_centers_)

    return torch.tensor(centroids_batch)


def manage_batch(gpu, batch, n_clusters, init, n_init, sample_points, dimensionality, n_classes):
    batch_size = int(batch.batch.size()[0] / sample_points)
    feature_matrix_batch = batch.pos.cuda(non_blocking=True).view(batch_size, sample_points, -1)
    noise = torch.normal(torch.zeros(batch_size, sample_points, dimensionality), torch.full((batch_size, sample_points,
            dimensionality), fill_value=0.1)).cuda(gpu)
    feature_matrix_batch = feature_matrix_batch + noise
    classes = batch.y.cuda(non_blocking=True).flatten()
    one_hot_encoding_batch = F.one_hot(classes, n_classes)
    struct_feature_matrix_batch = cluster(feature_matrix_batch, n_clusters, init, n_init).cuda(gpu)
    return feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch, classes


def draw_from_gen_std_class(gen_means,gen_stds,classes):
    N = gen_means.size()[0]
    rand = torch.randperm(N)
    rand = rand[:1]
    return gen_means[rand], gen_stds[rand], classes[rand]


def discriminator_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch):

    # Passing real instances
    model.get_submodule("module").is_discriminating = True
    real_critics = model(struct_feature_matrix_batch, one_hot_encoding_batch)
    d_real_loss = - torch.mean(real_critics)

    # Passing fake instances
    model.get_submodule("module").is_discriminating = False
    fake_critics, _, _, _ = model(feature_matrix_batch, one_hot_encoding_batch)
    d_fake_loss = torch.mean(fake_critics)

    # Discriminator loss
    d_loss = d_real_loss + d_fake_loss

    return d_loss


def discriminator_training(model, d_optimizer, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch, clamp_num):

    # Clear discriminator gradients
    d_optimizer.zero_grad()

    # Forward pass
    d_loss = discriminator_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch)

    # Optimise discriminator
    d_loss.backward()
    d_optimizer.step()

    # Clip discriminator's parameters
    for p in model.get_submodule("module.struct_discriminator").parameters():
        p.data.clamp_(-clamp_num, clamp_num)

    return d_loss


def encoder_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch):

    model.get_submodule("module").is_discriminating = False
    fake_critics, gen_struct_feature_matrix_batch, means_batch, log_stds_batch = model(feature_matrix_batch,
                                                                                       one_hot_encoding_batch)
    KLD = -0.5 * torch.sum(1 + log_stds_batch - means_batch.pow(2) - log_stds_batch.exp())
    #chamfer_distance = ChamferDistance()
    chamfer_loss, _ = chamfer_distance(struct_feature_matrix_batch, gen_struct_feature_matrix_batch)
    enc_loss = KLD + chamfer_loss + 0 * torch.mean(fake_critics)

    return means_batch, log_stds_batch, enc_loss


def encoder_training(model, enc_optimizer, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch):

    # Clear encoder gradients
    enc_optimizer.zero_grad()

    # Forward pass
    means_batch, log_stds_batch, enc_loss = encoder_forward_pass(model, feature_matrix_batch,
                                                                 struct_feature_matrix_batch, one_hot_encoding_batch)

    # Optimise encoder
    enc_loss.backward()
    enc_optimizer.step()

    return means_batch, log_stds_batch, enc_loss


def generator_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch):
    # Generate fake instances
    model.get_submodule("module").is_discriminating = False
    fake_critics, gen_struct_feature_matrix_batch, _, _ = model(feature_matrix_batch, one_hot_encoding_batch)
    gan_loss = - torch.mean(fake_critics)

    #chamfer_distance = ChamferDistance()
    chamfer_loss, _ = chamfer_distance(struct_feature_matrix_batch, gen_struct_feature_matrix_batch)

    g_loss = gan_loss + chamfer_loss * 0.1

    return g_loss


def generator_training(model, g_optimizer, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch):

    # Clear generator gradients
    g_optimizer.zero_grad()

    # Forward pass
    g_loss = generator_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch)

    # Optimise generator
    g_loss.backward()
    g_optimizer.step()

    return g_loss


# Training loop
def training_loop(gpu, gpus, training_dataloader, model, d_optimizer, enc_optimizer, g_optimizer, n_clusters, init, n_init, clamp_num, gen_trainining_ratio, sample_points, dimensionality, n_classes):
    d_losses = []
    enc_losses = []
    g_losses = []
    gen_training_batch = math.floor(len(training_dataloader) / (len(training_dataloader) / gen_trainining_ratio))  # all batch numbers which are multiple of this number will train the vae-generator as well
    gen_means = []
    gen_stds = []
    classes = []
    for batch_n, batch in enumerate(training_dataloader):
        # Manage batch
        feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch, classes_batch = manage_batch(gpu, batch, n_clusters, init, n_init, sample_points, dimensionality, n_classes)
        # Train discriminator
        d_loss = discriminator_training(model, d_optimizer, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch, clamp_num)
        d_losses.append(d_loss.item())

        enc_loss = None
        g_loss = None
        if batch_n % gen_training_batch == 0 and batch_n != 0:
            # Train encoder
            means_batch, log_stds_batch, enc_loss = encoder_training(model, enc_optimizer, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch)
            # Train generator
            g_loss = generator_training(model, g_optimizer, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch)

            enc_losses.append(enc_loss.item())
            g_losses.append(g_loss.item())
            # Append means, stds and classes
            gen_means.append(means_batch)
            stds_batch = torch.exp(0.5*log_stds_batch)
            gen_stds.append(stds_batch)
            classes.append(classes_batch)

        # Print losses for the batch
        dist.reduce(d_loss, 0)
        if enc_loss is not None:
            dist.reduce(enc_loss, 0)
            dist.reduce(g_loss, 0)
        if gpu == 0:
            print(f" Discriminator loss: {d_loss.item() / gpus:>7f}")
            if enc_loss is not None:
                print(f" Encoder loss: {enc_loss.item() / gpus:>7f}")
                print(f" Generator loss: {g_loss.item() / gpus:>7f}")

    return torch.cat(gen_means), torch.cat(gen_stds), torch.cat(classes), torch.tensor(d_losses).cuda(gpu),\
           torch.tensor(enc_losses).cuda(gpu), torch.tensor(g_losses).cuda(gpu)


# Test loop
def test_loop(gpu, test_dataloader, model, n_clusters, init, n_init, sample_points, dimensionality, n_classes):
    test_d_losses = []
    test_enc_losses = []
    test_g_losses = []

    with torch.no_grad():
        for batch_n, batch in enumerate(test_dataloader):

            # Manage batch
            feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch, classes_batch = manage_batch(gpu,
                            batch, n_clusters, init, n_init, sample_points, dimensionality, n_classes)

            # Discriminator forward pass
            d_loss = discriminator_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch)

            # Encoder forward pass
            _, _, enc_loss = encoder_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch)

            # VAE-Generator forward pass
            g_loss = generator_forward_pass(model, feature_matrix_batch, struct_feature_matrix_batch, one_hot_encoding_batch)

            # Append losses
            test_d_losses.append(d_loss.item())
            test_enc_losses.append(enc_loss.item())
            test_g_losses.append(g_loss.item())

    return torch.tensor(test_d_losses).cuda(gpu), torch.tensor(test_enc_losses).cuda(gpu),\
           torch.tensor(test_g_losses).cuda(gpu)


def generate_gaussian_mixture(gpu, triples, n_classes, gaussian_mixture):
    gen_means = []
    gen_stds = []
    classes = []

    for means, stds, cls in triples:
        gen_means.append(means)
        gen_stds.append(stds)
        classes.append(cls)

    gen_means = torch.cat(gen_means).cuda(gpu)
    gen_stds = torch.cat(gen_stds).cuda(gpu)
    classes = torch.cat(classes).cuda(gpu)

    for i in range(n_classes):
        index = classes == i
        if len(index) != 0:
            class_means = gen_means[index]
            class_stds = gen_stds[index]
            n_normal_dists = class_means.size()[0]
            mix = D.Categorical(torch.ones(n_normal_dists,).cuda(gpu))
            comp = D.Independent(D.Normal(class_means, class_stds), 1)
            gaussian_mixture[i] = D.MixtureSameFamily(mix, comp)


def generate_gaussian_mixture_gpu0(gpu, gen_means, gen_stds, classes, n_classes, gaussian_mixture):
    for i in range(n_classes):
        index = classes == i
        class_means = gen_means[index]
        if len(class_means) != 0:
            class_stds = gen_stds[index]
            n_normal_dists = class_means.size()[0]
            mix = D.Categorical(torch.ones(n_normal_dists, ).cuda(gpu))
            comp = D.Independent(D.Normal(class_means, class_stds), 1)
            gaussian_mixture[i] = D.MixtureSameFamily(mix, comp)


def train_optimisation(gpu, gpus, dir_path, start_epochs, end_epochs, training_dataloader, test_dataloader, model, d_optimizer, enc_optimizer, g_optimizer, d_scheduler, vae_g_scheduler, n_clusters, init, n_init, sample_points, dimensionality, n_classes, clamp_num, gen_training_ratio):
    avg_train_d_losses = []
    avg_train_enc_losses = []
    avg_train_g_losses = []
    avg_test_d_losses = []
    avg_test_enc_losses = []
    avg_test_g_losses = []
    d_learning_rates = []
    enc_learning_rates = []
    g_learning_rates = []
    gaussian_mixture = {}
    for i in range(start_epochs, end_epochs):
        if gpu == 0:
            print(f"Epoch {i + 1}\n-------------------------------")

        gen_means, gen_stds, classes, train_d_losses, train_enc_losses, train_g_losses = training_loop(gpu, gpus, training_dataloader, model, d_optimizer, enc_optimizer, g_optimizer, n_clusters, init, n_init, clamp_num, gen_training_ratio, sample_points, dimensionality,n_classes)
        #gen_means, gen_stds, classes = draw_from_gen_std_class(gen_means, gen_stds, classes)
        #output = [None for _ in range(gpus)]
        #dist.all_gather_object(output, (gen_means, gen_stds, classes))

        avg_train_d_loss = torch.mean(train_d_losses)
        avg_train_enc_loss = torch.mean(train_enc_losses)
        avg_train_g_loss = torch.mean(train_g_losses)
        dist.reduce(avg_train_d_loss, 0)
        dist.reduce(avg_train_enc_loss, 0)
        dist.reduce(avg_train_g_loss, 0)

        test_d_losses, test_enc_losses, test_g_losses = test_loop(gpu, test_dataloader, model, n_clusters, init, n_init, sample_points, dimensionality, n_classes)
        avg_test_d_loss = torch.mean(test_d_losses)
        avg_test_enc_loss = torch.mean(test_enc_losses)
        avg_test_g_loss = torch.mean(test_g_losses)
        dist.reduce(avg_test_d_loss, 0)
        dist.reduce(avg_test_enc_loss, 0)
        dist.reduce(avg_test_g_loss, 0)
        if gpu == 0:  # the following operations are performed only by the process running in the first gpu

            #generate_gaussian_mixture(gpu, output, n_classes, gaussian_mixture)
            generate_gaussian_mixture_gpu0(gpu, gen_means, gen_stds, classes, n_classes, gaussian_mixture)

            avg_train_d_loss = avg_train_d_loss / gpus  # average loss among all gpus
            avg_train_enc_loss = avg_train_enc_loss / gpus # average loss among all gpus
            avg_train_g_loss = avg_train_g_loss / gpus  # average loss among all gpus
            avg_test_d_loss = avg_test_d_loss / gpus  # average loss among all gpus
            avg_test_enc_loss = avg_test_enc_loss / gpus  # average loss among all gpus
            avg_test_g_loss = avg_test_g_loss / gpus  # average loss among all gpus

            # Append losses
            avg_train_d_losses.append(avg_train_d_loss.item())
            avg_train_enc_losses.append(avg_train_enc_loss.item())
            avg_train_g_losses.append(avg_train_g_loss.item())
            avg_test_d_losses.append(avg_test_d_loss.item())
            avg_test_enc_losses.append(avg_test_enc_loss.item())
            avg_test_g_losses.append(avg_test_g_loss.item())
            d_learning_rates.append((d_optimizer.param_groups[0])["lr"])
            enc_learning_rates.append((enc_optimizer.param_groups[0])["lr"])
            g_learning_rates.append((g_optimizer.param_groups[0])["lr"])

            print(f"Discriminator training average loss: {avg_train_d_loss.item()}")
            print(f"Encoder training average loss: {avg_train_enc_loss.item()}")
            print(f"Generator training average loss: {avg_train_g_loss.item()}")
            print(f"Discriminator test average loss: {avg_test_d_loss.item()}")
            print(f"Encoder test average loss: {avg_test_enc_loss.item()}")
            print(f"Generator test average loss: {avg_test_g_loss.item()}%")
            print(f"Discriminator learning rate: {(d_optimizer.param_groups[0])['lr']}%")
            print(f"Encoder learning rate: {(enc_optimizer.param_groups[0])['lr']}%")
            print(f"Generator learning rate: {(g_optimizer.param_groups[0])['lr']}%")

            save_checkpoint(i, model, d_optimizer, enc_optimizer, g_optimizer, d_scheduler, vae_g_scheduler, avg_train_d_losses,
                            avg_train_enc_losses, avg_train_g_losses, avg_test_d_losses, avg_test_enc_losses, avg_test_g_losses,
                            d_learning_rates, enc_learning_rates, g_learning_rates, gaussian_mixture,
                            os.path.join(dir_path, f"epoch{i}.pth"))

    return gaussian_mixture

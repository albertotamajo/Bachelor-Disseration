# Third-party modules
import torch.optim
import os
import torch.distributed as dist
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
#from chamferdist import ChamferDistance
from pytorch3d.loss import chamfer_distance


def save_checkpoint(epoch, model, d_optimizer, g_optimizer, d_scheduler, g_scheduler, train_d_losses,
                    train_g_losses, test_d_losses, test_g_losses, d_learning_rates, g_learning_rates, gaussian_mixture,
                    save_path):
    dict = {
        'epoch': epoch,
        'model_state_dict': model.get_submodule("module").state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'train_d_losses': train_d_losses,
        'train_g_losses': train_g_losses,
        'test_d_losses': test_d_losses,
        'test_g_losses': test_g_losses,
        'd_learning_rates': d_learning_rates,
        'g_learning_rates': g_learning_rates,
        'gaussian_mixture': gaussian_mixture
    }
    if d_scheduler is not None:
        dict['d_scheduler_state_dict'] = d_scheduler.state_dict()
    if g_scheduler is not None:
        dict['g_scheduler_state_dict'] = g_scheduler.state_dict()

    torch.save(dict, save_path)


def load_checkpoint(model, d_optimizer, g_optimizer, d_scheduler, g_scheduler, load_path):
    try:
        checkpoint = torch.load(load_path)
        print("Progress file in the folder")
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dictionary read")
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        print("Discriminator optimizer state dictionary read")
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        print("Generator optimizer state dictionary read")
        if d_scheduler is not None:
            d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
            print("Discriminator scheduler state dictionary read")
        if g_scheduler is not None:
            g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
            print("Generator scheduler state dictionary read")
        epoch = checkpoint['epoch']
        print("Epoch read")
        print(epoch + 1)
        return epoch + 1
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



def manage_batch(gpu, batch, sample_points, dimensionality, n_classes):
    batch_size = int(batch.batch.size()[0] / sample_points)
    feature_matrix_batch = batch.pos.cuda(non_blocking=True).view(batch_size, sample_points, -1)
    noise = torch.normal(torch.zeros(batch_size, sample_points, dimensionality), torch.full((batch_size, sample_points,
            dimensionality), fill_value=0.1)).cuda(gpu)
    feature_matrix_batch = feature_matrix_batch + noise
    classes = batch.y.cuda(non_blocking=True).flatten()
    one_hot_encoding_batch = F.one_hot(classes, n_classes)
    return feature_matrix_batch, one_hot_encoding_batch, classes


def generate_latent(gaussian_mixture, classes_batch):
    latent_vector_batch = []
    for i in classes_batch:
        latent_vector_batch.append(gaussian_mixture[i.item()].sample())

    return torch.stack(latent_vector_batch)


def discriminator_forward_pass(gpu, model, feature_matrix_batch, one_hot_encoding_batch, classes_batch, gaussian_mixture):

    # Passing real instances
    model.get_submodule("module").is_discriminating = True
    real_critics = model(feature_matrix_batch, one_hot_encoding_batch)
    d_real_loss = - torch.mean(real_critics)

    # Passing fake instances
    model.get_submodule("module").is_discriminating = False
    latent_vector_batch = generate_latent(gaussian_mixture, classes_batch).cuda(gpu)
    fake_critics, _, _ = model(latent_vector_batch, one_hot_encoding_batch)
    d_fake_loss = torch.mean(fake_critics)

    # Discriminator loss
    d_loss = d_real_loss + d_fake_loss

    return d_loss


def discriminator_training(gpu, model, d_optimizer, feature_matrix_batch, one_hot_encoding_batch, classes_batch, gaussian_mixture, clamp_num):

    # Clear discriminator gradients
    d_optimizer.zero_grad()

    # Forward pass
    d_loss = discriminator_forward_pass(gpu, model, feature_matrix_batch, one_hot_encoding_batch, classes_batch, gaussian_mixture)

    # Optimise discriminator
    d_loss.backward()
    d_optimizer.step()

    # Clip discriminator's parameters
    for p in model.get_submodule("module.final_discriminator").parameters():
        p.data.clamp_(-clamp_num, clamp_num)

    return d_loss


def generator_forward_pass(gpu, model, one_hot_encoding_batch, classes_batch, gaussian_mixture):

    # Generate fake instances
    model.get_submodule("module").is_discriminating = False
    latent_vector_batch = generate_latent(gaussian_mixture, classes_batch).cuda(gpu)
    fake_critics, struct_feature_matrix_batch, feature_matrix_batch = model(latent_vector_batch, one_hot_encoding_batch)
    gen_struct_feature_matrix_batch = cluster(feature_matrix_batch, struct_feature_matrix_batch.size()[1],
                                              "k-means++", 1).cuda(gpu)
    #chamfer_distance = ChamferDistance()
    chamfer_loss, _ = chamfer_distance(struct_feature_matrix_batch, gen_struct_feature_matrix_batch)
    g_loss = - torch.mean(fake_critics) + chamfer_loss * 0.1

    return g_loss


def generator_training(gpu, model, g_optimizer, one_hot_encoding_batch, classes_batch, gaussian_mixture):

    # Clear generator gradients
    g_optimizer.zero_grad()

    # Forward pass
    g_loss = generator_forward_pass(gpu, model, one_hot_encoding_batch, classes_batch, gaussian_mixture)

    # Optimise vae-generator
    g_loss.backward()
    g_optimizer.step()

    return g_loss


# Training loop
def training_loop(gpu, gpus, training_dataloader, model, d_optimizer, g_optimizer, clamp_num, gen_trainining_ratio, gaussian_mixture, sample_points, dimensionality, n_classes):
    d_losses = []
    g_losses = []
    gen_training_batch = math.floor(len(training_dataloader) / (len(training_dataloader) / gen_trainining_ratio))  # all batch numbers which are multiple of this number will train the vae-generator as well
    for batch_n, batch in enumerate(training_dataloader):
        # Manage batch
        feature_matrix_batch, one_hot_encoding_batch, classes_batch = manage_batch(gpu, batch, sample_points, dimensionality, n_classes)
        # Train discriminator
        d_loss = discriminator_training(gpu, model, d_optimizer, feature_matrix_batch, one_hot_encoding_batch, classes_batch, gaussian_mixture, clamp_num)
        d_losses.append(d_loss.item())

        g_loss = None
        if batch_n % gen_training_batch == 0 and batch_n != 0:
            # Train generator
            g_loss = generator_training(gpu, model, g_optimizer, one_hot_encoding_batch, classes_batch, gaussian_mixture)
            g_losses.append(g_loss.item())


        # Print losses for the batch
        dist.reduce(d_loss, 0)
        if g_loss is not None:
            dist.reduce(g_loss, 0)
        if gpu == 0:
            print(f" Discriminator loss: {d_loss.item() / gpus:>7f}")
            if g_loss is not None:
                print(f" Generator loss: {g_loss.item() / gpus:>7f}")

    return torch.tensor(d_losses).cuda(gpu), torch.tensor(g_losses).cuda(gpu)


# Test loop
def test_loop(gpu, test_dataloader, model, gaussian_matrix, sample_points, dimensionality, n_classes):
    test_d_losses = []
    test_g_losses = []

    with torch.no_grad():
        for batch_n, batch in enumerate(test_dataloader):

            # Manage batch
            feature_matrix_batch, one_hot_encoding_batch, classes_batch = manage_batch(gpu, batch, sample_points, dimensionality, n_classes)

            # Discriminator forward pass
            d_loss = discriminator_forward_pass(gpu, model, feature_matrix_batch, one_hot_encoding_batch, classes_batch, gaussian_matrix)

            # Generator forward pass
            g_loss = generator_forward_pass(gpu, model, one_hot_encoding_batch, classes_batch, gaussian_matrix)

            # Append losses
            test_d_losses.append(d_loss.item())
            test_g_losses.append(g_loss.item())

    return torch.tensor(test_d_losses).cuda(gpu), torch.tensor(test_g_losses).cuda(gpu)


def train_optimisation(gpu, gpus, dir_path, start_epochs, end_epochs, training_dataloader, test_dataloader, model, d_optimizer, g_optimizer, d_scheduler, g_scheduler, clamp_num, gen_training_ratio, gaussian_mixture, sample_points, dimensionality, n_classes):
    avg_train_d_losses = []
    avg_train_g_losses = []
    avg_test_d_losses = []
    avg_test_g_losses = []
    d_learning_rates = []
    g_learning_rates = []
    for i in range(start_epochs, end_epochs):
        if gpu == 0:
            print(f"Epoch {i + 1}\n-------------------------------")

        train_d_losses, train_g_losses = training_loop(gpu, gpus, training_dataloader, model, d_optimizer, g_optimizer, clamp_num, gen_training_ratio, gaussian_mixture, sample_points, dimensionality, n_classes)
        avg_train_d_loss = torch.mean(train_d_losses)
        avg_train_g_loss = torch.mean(train_g_losses)
        dist.reduce(avg_train_d_loss, 0)
        dist.reduce(avg_train_g_loss, 0)

        test_d_losses, test_g_losses = test_loop(gpu, test_dataloader, model, gaussian_mixture, sample_points, dimensionality, n_classes)
        avg_test_d_loss = torch.mean(test_d_losses)
        avg_test_g_loss = torch.mean(test_g_losses)
        dist.reduce(avg_test_d_loss, 0)
        dist.reduce(avg_test_g_loss, 0)
        if gpu == 0:  # the following operations are performed only by the process running in the first gpu

            avg_train_d_loss = avg_train_d_loss / gpus  # average loss among all gpus
            avg_train_g_loss = avg_train_g_loss / gpus  # average loss among all gpus
            avg_test_d_loss = avg_test_d_loss / gpus  # average loss among all gpus
            avg_test_g_loss = avg_test_g_loss / gpus  # average loss among all gpus

            # Append losses
            avg_train_d_losses.append(avg_train_d_loss.item())
            avg_train_g_losses.append(avg_train_g_loss.item())
            avg_test_d_losses.append(avg_test_d_loss.item())
            avg_test_g_losses.append(avg_test_g_loss.item())
            d_learning_rates.append((d_optimizer.param_groups[0])["lr"])
            g_learning_rates.append((g_optimizer.param_groups[0])["lr"])

            print(f"Discriminator training average loss: {avg_train_d_loss.item()}")
            print(f"Generator training average loss: {avg_train_g_loss.item()}")
            print(f"Discriminator test average loss: {avg_test_d_loss.item()}")
            print(f"Generator test average loss: {avg_test_g_loss.item()}%")
            print(f"Discriminator learning rate: {(d_optimizer.param_groups[0])['lr']}%")
            print(f"Generator learning rate: {(g_optimizer.param_groups[0])['lr']}%")

            save_checkpoint(i, model, d_optimizer, g_optimizer, d_scheduler, g_scheduler, avg_train_d_losses,
                            avg_train_g_losses, avg_test_d_losses, avg_test_g_losses, d_learning_rates, g_learning_rates,
                            gaussian_mixture, os.path.join(dir_path, f"epoch{i}.pth"))
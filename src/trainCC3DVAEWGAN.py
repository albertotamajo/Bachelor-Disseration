# Built-in modules
from vaegan import Struct_VAE_GAN
from finalgan import FinalGAN
import trainStructVAEGAN
import trainFinalVAEGAN
import encoderArchitecture as encoderArchi
import structureGeneratorArchitecture as structGenArchi
import structureDiscrminatorArchitecture as structDisArchi
import finalDiscriminatorArchitecture as finalDisArchi
import finalGeneratorArchitecture as finalGenArchi

# Third-party modules
import torch.optim
from torch_geometric.datasets import ModelNet
import torch_geometric.data as data
from torch_geometric.transforms import SamplePoints, NormalizeScale
from torch.utils.data.distributed import DistributedSampler
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from xml.etree import ElementTree as ET
import argparse
from itertools import chain

torch.set_printoptions(threshold=10_000)
torch.autograd.set_detect_anomaly(True)

# -------------------------------------BEGINNING STRUCTURE PART---------------------------------------------- #

# Encoder
encoder = encoderArchi.encoder

# Structure Generator
struct_generator = structGenArchi.struct_generator

# CREATE Structure Discriminator
struct_discriminator = structDisArchi.struct_discriminator

# CREATE Structure VAE-GAN
struct_vae_gan = Struct_VAE_GAN(encoder, struct_generator, struct_discriminator)

# -----------------------------------------END STRUCTURE PART------------------------------------------------ #


# -----------------------------------------BEGINNING FINAL PART---------------------------------------------- #


# CREATE Final Generator
final_generator = finalGenArchi.final_generator

# Final Discriminator
final_discriminator = finalDisArchi.final_discriminator

# CREATE Final GAN
final_gan = FinalGAN(struct_generator, final_generator, final_discriminator)


# -------------------------------------------END FINAL PART-------------------------------------------------- #


# ---------------------------------------------FUNCTIONS-------------------------------------------------------------- #

def createdir(path):
    try:
        os.mkdir(path)
        print(f"Directory '{path}' created")
    except FileExistsError:
        print(f"Directory '{path}' already exists")


def next_training_number(path):
    try:
        listdir = os.listdir(path)
        if listdir == []:
            return 1
        else:
            list_number = map(lambda x: int(x.replace("train", "")), filter(lambda x: x.startswith("train"), listdir))
            return max(list_number) + 1 if list_number is not [] else 1
    except:
        return 1


def parse_xml(file):
    return ET.parse(file).getroot()


def parse_training_options(root):
    train_stage = int(root.find('train').text)
    save_path = root.find('save').text
    load_path = root.find('load').text
    return train_stage, save_path, load_path


def parse_struct_vae_gan_xml(root):
    struct_sample_points = None
    struct_points = None
    struct_clamp_num = None
    struct_gen_training_ratio = None
    struct_k_means_init = None
    struct_k_means_iter = None
    struct_batch_size = None
    struct_dimensionality = None
    struct_d_learning_rate = None
    struct_enc_learning_rate = None
    struct_g_learning_rate = None
    struct_start_epochs = None
    struct_end_epochs = None
    struct_n_classes = None
    for struct in root.findall("struct_vaegan"):
        struct_sample_points = int(struct.find('sample_points').text)
        struct_points = int(struct.find('blueprint_points').text)
        struct_clamp_num = float(struct.find('clamp_num').text)
        struct_gen_training_ratio = int(struct.find('gen_training_ratio').text)
        struct_k_means_init = struct.find('k_means_init').text
        struct_k_means_iter = int(struct.find('k_means_iter').text)
        struct_batch_size = int(struct.find('batch_size').text)
        struct_dimensionality = int(struct.find('dimensionality').text)
        struct_d_learning_rate = float(struct.find('d_learning_rate').text)
        struct_enc_learning_rate = float(struct.find('enc_learning_rate').text)
        struct_g_learning_rate = float(struct.find('g_learning_rate').text)
        struct_start_epochs = int(struct.find('start_epochs').text)
        struct_end_epochs = int(struct.find('end_epochs').text)
        struct_n_classes = int(struct.find('n_classes').text)

    return struct_sample_points, struct_points, struct_clamp_num, struct_gen_training_ratio, struct_k_means_init,\
           struct_k_means_iter, struct_batch_size, struct_dimensionality, struct_d_learning_rate,\
           struct_enc_learning_rate, struct_g_learning_rate, struct_start_epochs, struct_end_epochs, struct_n_classes


def parse_final_gan_xml(root):
    final_sample_points = None
    final_clamp_num = None
    final_gen_training_ratio = None
    final_dimensionality = None
    final_batch_size = None
    final_d_learning_rate = None
    final_g_learning_rate = None
    final_start_epochs = None
    final_end_epochs = None
    final_n_classes = None
    for final in root.findall("final_gan"):
        final_sample_points = int(final.find('sample_points').text)
        final_clamp_num = float(final.find('clamp_num').text)
        final_gen_training_ratio = int(final.find('gen_training_ratio').text)
        final_dimensionality = int(final.find('dimensionality').text)
        final_batch_size = int(final.find('batch_size').text)
        final_d_learning_rate = float(final.find('d_learning_rate').text)
        final_g_learning_rate = float(final.find('g_learning_rate').text)
        final_start_epochs = int(final.find('start_epochs').text)
        final_end_epochs = int(final.find('end_epochs').text)
        final_n_classes = int(final.find('n_classes').text)

    return final_sample_points, final_clamp_num, final_gen_training_ratio, final_dimensionality, final_batch_size,\
           final_d_learning_rate, final_g_learning_rate, final_start_epochs, final_end_epochs, final_n_classes


def train_struct_vaegan(gpu, gpus, world_size, struct_vae_gan, struct_path, hyperparameters, load_path=None):

    struct_sample_points, struct_points, struct_clamp_num, struct_gen_training_ratio, struct_k_means_init, \
    struct_k_means_iter, struct_batch_size, struct_dimensionality, struct_d_learning_rate, \
    struct_enc_learning_rate, struct_g_learning_rate, struct_start_epochs, struct_end_epochs, struct_n_classes = hyperparameters


    training_data = ModelNet("ModelNet10_train_data",
                             transform=lambda x: NormalizeScale()(SamplePoints(struct_sample_points)(x)))
    training_sampler = DistributedSampler(training_data, num_replicas=world_size)
    training_dataloader = data.DataLoader(dataset=training_data, batch_size=struct_batch_size, shuffle=False,
                                          num_workers=0,
                                          pin_memory=True, sampler=training_sampler)

    test_data = ModelNet("Modelnet10_test_data", train=False,
                         transform=lambda x: NormalizeScale()(SamplePoints(struct_sample_points)(x)))
    test_sampler = DistributedSampler(test_data, num_replicas=world_size)
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=struct_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True, sampler=test_sampler)

    struct_vae_gan.cuda(gpu)
    d_optimizer = torch.optim.RMSprop(struct_vae_gan.get_submodule("struct_discriminator").parameters(),
                                    struct_d_learning_rate)

    enc_optimizer = torch.optim.RMSprop(struct_vae_gan.get_submodule("encoder").parameters(), struct_enc_learning_rate)
    g_optimizer = torch.optim.RMSprop(struct_vae_gan.get_submodule("struct_generator").parameters(),
                                      struct_g_learning_rate)

    if load_path is not None:
        trainStructVAEGAN.load_checkpoint(struct_vae_gan, d_optimizer, enc_optimizer, g_optimizer, None, None, load_path)

    struct_vae_gan = DDP(struct_vae_gan, device_ids=[gpu])

    gaussian_mixture = trainStructVAEGAN.train_optimisation(gpu, gpus, struct_path, struct_start_epochs,
                                                            struct_end_epochs, training_dataloader,
                                                            test_dataloader, struct_vae_gan, d_optimizer,
                                                            enc_optimizer, g_optimizer, None, None, struct_points,
                                                            struct_k_means_init, struct_k_means_iter,
                                                            struct_sample_points, struct_dimensionality,
                                                            struct_n_classes, struct_clamp_num,
                                                            struct_gen_training_ratio)
    return gaussian_mixture


def train_finalgan(gpu, gpus, world_size, final_gan, final_path, gaussian_mixture, hyperparameters, load_path=None):

    final_sample_points, final_clamp_num, final_gen_training_ratio, final_dimensionality, final_batch_size, \
    final_d_learning_rate, final_g_learning_rate, final_start_epochs, final_end_epochs, final_n_classes = hyperparameters

    training_data = ModelNet("ModelNet10_train_data",
                             transform=lambda x: NormalizeScale()(SamplePoints(final_sample_points)(x)))
    training_sampler = DistributedSampler(training_data, num_replicas=world_size)
    training_dataloader = data.DataLoader(dataset=training_data, batch_size=final_batch_size, shuffle=False,
                                          num_workers=0,
                                          pin_memory=True, sampler=training_sampler)

    test_data = ModelNet("Modelnet10_test_data", train=False,
                         transform=lambda x: NormalizeScale()(SamplePoints(final_sample_points)(x)))
    test_sampler = DistributedSampler(test_data, num_replicas=world_size)
    test_dataloader = data.DataLoader(dataset=test_data, batch_size=final_batch_size, shuffle=False, num_workers=0,
                                      pin_memory=True, sampler=test_sampler)

    final_gan.cuda(gpu)
    d_optimizer = torch.optim.RMSprop(final_gan.get_submodule("final_discriminator").parameters(),
                                        final_d_learning_rate)
    g_optimizer = torch.optim.RMSprop(final_gan.get_submodule("final_generator").parameters(), final_g_learning_rate)

    if load_path is not None:
        trainFinalVAEGAN.load_checkpoint(final_gan, d_optimizer, g_optimizer, None, None, load_path)

    final_gan = DDP(final_gan, device_ids=[gpu])

    trainFinalVAEGAN.train_optimisation(gpu, gpus, final_path, final_start_epochs, final_end_epochs, training_dataloader,
                                        test_dataloader, final_gan, d_optimizer, g_optimizer, None, None,
                                        final_clamp_num, final_gen_training_ratio,
                                        gaussian_mixture, final_sample_points, final_dimensionality, final_n_classes)


# ---------------------------------------------FUNCTIONS-------------------------------------------------------------- #


# -----------------------------------------BEGINNING MULTI-GPU TRAINING----------------------------------------------- #

def train(args, gpus, world_size, struct_vae_gan, final_gan):
    gpu = args.local_rank
    root = parse_xml(args.hyperparam)
    train_stage, save_path, load_path = parse_training_options(root)
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(gpu)
    if gpu == 0:
        print(f"GPUs: {gpus}")
    if train_stage == 0:
        struct_path = None
        final_path = None
        if gpu == 0:
            dir_path = os.path.join("cc3dvaewgan", f"train{next_training_number('cc3dvaewgan')}") if save_path is None else save_path
            createdir(dir_path)
            struct_path = os.path.join(dir_path, "blueprintVAEWGAN")
            createdir(struct_path)
            final_path = os.path.join(dir_path, "finalWGAN")
            createdir(final_path)
        gaussian_mixture = train_struct_vaegan(gpu, gpus, world_size, struct_vae_gan, struct_path,
                                                parse_struct_vae_gan_xml(root), load_path=load_path)
        train_finalgan(gpu, gpus, world_size, final_gan, final_path, gaussian_mixture, parse_final_gan_xml(root),
                       load_path=load_path)

    elif train_stage == 1:
        final_path = None
        if gpu == 0:
            final_path = os.path.join(save_path, "finalWGAN")
            createdir(final_path)
        gaussian_mixture = trainStructVAEGAN.load_gaussian_mixture(load_path)
        train_finalgan(gpu, gpus, world_size, final_gan, final_path, gaussian_mixture, parse_final_gan_xml(root))
    else:
        final_path = None
        if gpu == 0:
            final_path = os.path.join(save_path, "finalWGAN")
            createdir(final_path)
        gaussian_mixture = trainFinalVAEGAN.load_gaussian_mixture(load_path)
        train_finalgan(gpu, gpus, world_size, final_gan, final_path, gaussian_mixture, parse_final_gan_xml(root), load_path)


# -----------------------------------------END MULTI-GPU TRAINING----------------------------------------------------- #


if __name__ == '__main__':
    gpus = torch.cuda.device_count()
    gpus = int(gpus)
    nodes = 1
    world_size = nodes * gpus
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--hyperparam", type=str, required=True)  # load hyperparameters from xml file
    args = parser.parse_args()
    train(args, gpus, world_size, struct_vae_gan, final_gan)
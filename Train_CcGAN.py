import torch
import numpy as np
import os
import timeit
import config
import whitebox_utils
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
import itertools
from surface_extractor import GuidedFilter
from texture_extractor import ColorShift

from utils import *
from opts import parse_opts

from torch.utils.tensorboard import SummaryWriter

sw = SummaryWriter(config.PROJECT_NAME)

''' Settings '''
args = parse_opts()
config.DEVICE = args.cuda
config.PROJECT_NAME = args.project_name
config.BATCH_SIZE = args.batch_size_disc
config.LAMBDA_SURFACE = args.lambda_surface
config.LAMBDA_TEXTURE = args.lambda_texture
config.LAMBDA_CONTENT = args.lambda_context

# some parameters in opts
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_gan = args.dim_gan
lr_g = args.lr_g_gan
lr_d = args.lr_d_gan
save_niters_freq = args.save_niters_freq
batch_size_disc = args.batch_size_disc
batch_size_gene = args.batch_size_gene
batch_size_max = max(batch_size_disc, batch_size_gene)

threshold_type = args.threshold_type
nonzero_soft_weight_threshold = args.nonzero_soft_weight_threshold

NC = args.num_channels
IMG_SIZE = args.img_size

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]),
])

def pretrain_gen(netG, net_y2h_surface, opt_gen, photos, l1_loss, VGG):
    for epoch in range(0, config.NUM_PRETRAIN_EPOCHS):
        z = photos[np.random.choice(photos.shape[0], batch_size_max)]
        tmp = []
        for idx in range(0, batch_size_max):
            tmp.append(preprocess(z[idx]).unsqueeze(0))
        z = torch.cat(tmp, dim=0).to(config.DEVICE)

        batch_fake_images = netG(
            z, 
            net_y2h_surface(torch.rand(batch_size_max).to(config.DEVICE)), 
        )
        photos_vgg = VGG(z)
        fake_vgg = VGG(batch_fake_images)
        reconstruction_loss = l1_loss(photos_vgg, fake_vgg) * 255
        
        opt_gen.zero_grad()
        reconstruction_loss.backward()
        opt_gen.step()
        if((epoch+1)%100 == 0):
            print('[%d/%d] - Recon loss: %.8f' % ((epoch + 1), config.NUM_PRETRAIN_EPOCHS, reconstruction_loss.item()))

def train_CcGAN(kernel_sigma, kappa, photos, train_images, train_labels, gen, disc_surface, net_y2h_surface, VGG, save_images_folder, save_models_folder = None, clip_label=False):
    '''
    Note that train_images are not normalized to [-1,1]
    '''
    l1_loss = nn.L1Loss()    

    gen = gen.to(config.DEVICE)
    disc_surface = disc_surface.to(config.DEVICE)

    optimizerG = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(disc_surface.parameters(), lr=lr_d, betas=(0.5, 0.999))

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/CcGAN_{}_checkpoint_intrain/CcGAN_checkpoint_niters_{}.pth".format(threshold_type, resume_niters)
        checkpoint = torch.load(save_file)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc_surface.load_state_dict(checkpoint['disc_surface_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        print("load generator from", save_file)
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        print("load discriminator from", save_file)
        
        torch.set_rng_state(checkpoint['rng_state'])
    elif os.path.isfile(save_models_folder + "/CcGAN_pretrained_gen.pth"):
        save_file = save_models_folder + "/CcGAN_pretrained_gen.pth"
        checkpoint = torch.load(save_file)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        print("load generator from", save_models_folder + "/CcGAN_pretrained_gen.pth")
    else:
        pretrain_gen(gen, net_y2h_surface, optimizerG, photos, l1_loss, VGG)
        save_file = save_models_folder + "/CcGAN_pretrained_gen.pth"
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        torch.save({
            'gen_state_dict': gen.state_dict(),
            'disc_surface_state_dict': disc_surface.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'rng_state': torch.get_rng_state()
        }, save_file)

    #################
    unique_train_labels = np.sort(np.array(list(set(train_labels))))

    # printed images with labels between the 5-th quantile and 95-th quantile of training labels
    n_row=4; n_col = n_row
    
    start_label = np.quantile(train_labels, 0.05)
    end_label = np.quantile(train_labels, 0.95)
    selected_labels = np.linspace(start_label, end_label, num=n_row)
    y_fixed = np.zeros(n_row)
    for i in range(n_row):
        y_fixed[i] = selected_labels[i]
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).to(config.DEVICE)

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        '''  Train Discriminator   '''
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_surface_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_max, replace=True)
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_max)
        batch_target_surface_labels_with_epsilon = batch_target_surface_labels_in_dataset + batch_epsilons
        if clip_label:
            batch_target_surface_labels_with_epsilon = np.clip(batch_target_surface_labels_with_epsilon, 0.0, 1.0)

        batch_target_surface_labels = batch_target_surface_labels_with_epsilon[0:batch_size_disc]

        ## find index of real images with labels in the vicinity of batch_target_labels
        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
        batch_real_surface_indx = np.zeros(batch_size_disc, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
        batch_fake_surface_labels = np.zeros(batch_size_disc)

        for j in range(batch_size_disc):
            ## index for real images
            if threshold_type == "hard":
                assert False 
                # indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
            else:
                # reverse the weight function for SVDL
                indx_real_surface_in_vicinity = np.where((train_labels-batch_target_surface_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

            ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
            while len(indx_real_surface_in_vicinity)<1:
                batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                batch_target_surface_labels[j] = batch_target_surface_labels_in_dataset[j] + batch_epsilons_j
                if clip_label:
                    batch_target_surface_labels = np.clip(batch_target_surface_labels, 0.0, 1.0)
                ## index for real images
                if threshold_type == "hard":
                    # indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
                    assert False
                else:
                    # reverse the weight function for SVDL
                    indx_real_surface_in_vicinity = np.where((train_labels-batch_target_surface_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

            assert len(indx_real_surface_in_vicinity)>=1

            batch_real_surface_indx[j] = np.random.choice(indx_real_surface_in_vicinity, size=1)[0]

            ## labels for fake images generation
            if threshold_type == "hard":
                assert False
            else:
                lb_surface = batch_target_surface_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                ub_surface = batch_target_surface_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
            lb_surface = max(0.0, lb_surface); ub_surface = min(ub_surface, 1.0)
            assert lb_surface<=ub_surface
            assert lb_surface>=0 and ub_surface>=0
            assert lb_surface<=1 and ub_surface<=1
            batch_fake_surface_labels[j] = np.random.uniform(lb_surface, ub_surface, size=1)[0]
        #end for j

        ## draw the real image batch from the training set
        batch_real_surface_images = train_images[batch_real_surface_indx]
        
        assert batch_real_surface_images.max()>1

        batch_real_surface_labels = train_labels[batch_real_surface_indx]
        batch_real_surface_labels = torch.from_numpy(batch_real_surface_labels).type(torch.float).to(config.DEVICE)

        ## normalize real images to [-1,1]
        tmp = []
        for idx in range(0, batch_real_surface_images.shape[0]):
            tmp.append(preprocess(batch_real_surface_images[idx]).unsqueeze(0))
        batch_real_surface_images = torch.cat(tmp, dim=0).to(config.DEVICE)


        ## generate the fake image batch
        batch_fake_surface_labels = torch.from_numpy(batch_fake_surface_labels).type(torch.float).to(config.DEVICE)

        z = photos[np.random.choice(photos.shape[0], batch_size_max)]
        tmp = []
        for idx in range(0, batch_size_max):
            tmp.append(preprocess(z[idx]).unsqueeze(0))
        z = torch.cat(tmp, dim=0).to(config.DEVICE)
        batch_fake_images = gen(z, net_y2h_surface(batch_fake_surface_labels))

        ## target labels on gpu
        batch_target_surface_labels = torch.from_numpy(batch_target_surface_labels).type(torch.float).to(config.DEVICE)

        ## weight vector
        if threshold_type == "soft":
            real_surface_weights = torch.exp(-kappa*(batch_real_surface_labels - batch_target_surface_labels)**2).to(config.DEVICE)
            fake_surface_weights = torch.exp(-kappa*(batch_fake_surface_labels - batch_target_surface_labels)**2).to(config.DEVICE)
        else:
            assert False
            # real_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
            # fake_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
        #end if threshold type

        # forward pass
        dis_real_surface = disc_surface(
            batch_real_surface_images, 
            net_y2h_surface(batch_target_surface_labels)
        )
        dis_fake_surface = disc_surface(
            batch_fake_images, 
            net_y2h_surface(batch_target_surface_labels)
        )

        # use hinge loss type
        d_loss_real_surface = torch.nn.ReLU()(1.0 - dis_real_surface)
        d_loss_fake_surface = torch.nn.ReLU()(1.0 + dis_fake_surface)

        d_surface_loss = torch.mean(real_surface_weights.view(-1) * d_loss_real_surface.view(-1)) + torch.mean(fake_surface_weights.view(-1) * d_loss_fake_surface.view(-1))

        d_loss = d_surface_loss

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()


        '''  Train Generator   '''
        gen.train()

        # generate fake images
        batch_target_surface_labels = batch_target_surface_labels_with_epsilon[0:batch_size_gene]
        batch_target_surface_labels = torch.from_numpy(batch_target_surface_labels).type(torch.float).to(config.DEVICE)

        photo_sample_indices = np.random.randint(0, photos.shape[0], (batch_size_max, )) # N, 256, 256, 3
        sampled_photos = photos[photo_sample_indices]

        tmp = []
        for idx in range(0, sampled_photos.shape[0]):
            tmp.append(preprocess(sampled_photos[idx]).unsqueeze(0))
        z = torch.cat(tmp, dim = 0).to(config.DEVICE)

        batch_fake_images = gen(z, net_y2h_surface(batch_target_surface_labels))
        
        fake_image_vgg = VGG(batch_fake_images)
        real_image_vgg = VGG(z)

        content_loss = config.LAMBDA_CONTENT * l1_loss(fake_image_vgg, real_image_vgg)

        # loss
        dis_surface = disc_surface(
            batch_fake_images,
            net_y2h_surface(batch_target_surface_labels)
        )

        # still, use hinge loss_type
        g_surface_loss = - config.LAMBDA_SURFACE * dis_surface.mean()

        g_loss = g_surface_loss + content_loss

        # backward
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        # print loss
        if (niter+1) % 25 == 0:
            sw.add_scalar("D loss", d_loss.item(), niter)
            sw.add_scalar("G loss", g_loss.item(), niter)
            sw.add_scalar("G like loss", g_surface_loss.item(), niter)
            sw.add_scalar("content loss", content_loss.item(), niter)
            print (config.PROJECT_NAME + " CcGAN: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), timeit.default_timer()-start_time))
            print ("[dis_real_surface: %.3f] [dis_fake_surface: %.3f]" % 
                    (dis_real_surface.mean().item(), dis_fake_surface.mean().item()))
            print ("[D like loss: %.4e] [G like loss: %.4e] [content loss: %.4e]" % 
                    (d_surface_loss.item(), g_surface_loss.item(), content_loss.item()))

        if (niter+1) % 100 == 0:
            gen.eval()
            with torch.no_grad():
                photo_sample_indices = np.random.randint(0, photos.shape[0], (1, )) # 2, 256, 256, 3
                test_photo = photos[photo_sample_indices]
                tmp = []
                for k in range(0, test_photo.shape[0]):
                    tmp.append(preprocess(test_photo[k]).unsqueeze(0))
                z_test = torch.cat(tmp, dim=0).to(config.DEVICE)
                tmp = []
                for k in range(0, y_fixed.shape[0]):
                    gen_img = gen(z_test, net_y2h_surface(y_fixed[k]))
                    gen_img = gen_img[0:1, :, :, :]
                    tmp.append(torch.cat((z_test[0:1,:,:,:], gen_img), axis=3))
                gen_imgs = torch.cat(tmp, dim=0)
                gen_imgs = gen_imgs.detach().cpu()
                save_image(gen_imgs*0.5+0.5, save_images_folder +'/{}.png'.format(niter+1), nrow=n_row, normalize=True)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/CcGAN_{}_checkpoint_intrain/CcGAN_checkpoint_niters_{}.pth".format(threshold_type, niter+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'gen_state_dict': gen.state_dict(),
                    'disc_surface_state_dict': disc_surface.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for niter
    return gen, disc_surface


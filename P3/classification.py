'''
Start code for Project 3
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Prameth Gaddale
    PSU Email ID: pqg5273@psu.edu
    Description:
        test: Test the model.
        train: Train the model and periodically log the loss and accuracy.\
        visualize_layer: Visualize the activations of a layer in the model.
        visualize_tsne: Visualize the predictions using t-SNE.
        wallpaper_main: Main function for the wallpaper classification task.
            Put the --baseline argument to use the baseline model.
            Put the --improved argument to use the improved model.
        taiji_main: Main function for the taiji classification task.
            Put the --baseline argument to use the baseline model.
            Put the --improved argument to use the improved model.
}
'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset

from util import *
from models import *

def test(model, test_loader, device, criterion):
    """
    Test the model.
    Args:
        model (torch.nn.Module): Model to test.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        device (torch.device): Device to use (cuda or cpu).
        criterion (torch.nn.Module): Loss function to use.
    Returns:
        test_loss (float): Average loss on the test set.
        test_acc (float): Average accuracy on the test set.
        preds (numpy array): Class predictions.
        targets (numpy array): Target values.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        preds = []
        targets = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            targets.append(target.cpu().numpy())
            preds.append(pred.cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return test_loss, test_acc, preds, targets

#  Note log_interval doesn't actually log to a file but is used for printing. This can be changed if you want to log to a file.
def train(model, train_loader, optimizer, criterion, epochs, 
          log_interval, device):
    """
    Train the model and periodically log the loss and accuracy.
    Args:
        model (torch.nn.Module): Model to train.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer to use.
        criterion (torch.nn.Module): Loss function to use.
        epochs (int): Number of epochs to train for.
        log_interval (int): Print loss every log_interval epochs.
        device (torch.device): Device to use (cuda or cpu).
    Returns:
        per_epoch_loss (list): List of loss values per epoch.
        per_epoch_acc (list): List of accuracy values per epoch.
    """
    model.train()
    per_epoch_loss = []
    per_epoch_acc = []
    for epoch in range(epochs):
        train_loss = 0
        preds = []
        targets = []
        correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Get the accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Save the predictions and targets if it's the last epoch
            if epoch == epochs - 1:
                preds.append(pred.cpu().numpy())
                targets.append(target.cpu().numpy())
        train_loss /= len(train_loader)
        train_acc = correct / len(train_loader.dataset)
        per_epoch_acc.append(train_acc)
        if epoch % log_interval == 0:
            print('Epoch: {}, Loss: {}, Acc: {}'.format(epoch, train_loss, train_acc))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return model, np.array(per_epoch_loss), np.array(per_epoch_acc), preds, targets

def visualize_layer(model, loader, num_layer = 1, device = "cpu", filename = "layer.png"):
    """
    Visualize the activations of a layer in the model.
    Args:
        model (torch.nn.Module): Model to visualize.
        loader (torch.utils.data.DataLoader): Data loader to use.
        layer (int): Layer to visualize.
        device (torch.device): Device to use (cuda or cpu).
    """
    # Model is put in eval mode so that dropout and batchnorm layers are not activated
    model.eval()
    # Label names
    labels = ["CM", "CMM", "P1", "P2", "P3", "P31M", "P2M1", "P4", "P4G", "P4M", "P6", "P6M", "PG", "PGG", "PM", "PMG", "PMM"]
    outputs = []
    # Layer-wise activations are stored in outputs
    for i in range(0, len(loader.dataset), len(loader.dataset)//17):
        layer_out = []
        img = torch.unsqueeze(loader.dataset[i][0].to(device), 0)
        for layer in list(model.children())[0]:
            img = layer(img)
            layer_out.append(img)
        outputs.append(layer_out)
    # Plot the activations
    fig = plt.figure(figsize = (12, 12))
    for i in range(len(outputs)):
        img = torch.mean(torch.squeeze(outputs[i][num_layer], 0), 0).cpu().detach().numpy()
        plt.subplot(4, 5, i+1)
        plt.imshow(img, cmap = "gray")
        plt.title(labels[i])
    # Save the plot
    plt.savefig(filename, bbox_inches='tight')

def normalize(x):
    """
    Normalize the features to [0,1] range.
    Args:
        x (numpy array): Features to normalize.
    Returns:
        x (numpy array): Normalized features.
    """
    range_values = (np.max(x) - np.min(x))
    shift = x - np.min(x)
    x = shift / range_values
    return x

def visualize_tsne(model, loader, device = "cpu", filename = "tsne.png"):
    """
    Visualize the predictions using t-SNE.
    Args:
        dataset (torch.utils.data.Dataset): Dataset to visualize.
        filename (str): Filename to save the visualization to.
    """
    features = torch.tensor([])
    labels = []
    # Model is put in eval mode so that dropout and batchnorm layers are not activated
    model.eval()
    # Get the features and labels
    for batch in loader:
        images = batch[0].to(device)
        labels += batch[1].tolist()
        output = model.forward(images).cpu().detach()
        features = torch.cat((features, output), 0)
    # Normalize the features and apply t-SNE
    tsne = TSNE(n_components=2, random_state=0).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    tx = normalize(tx)
    ty = normalize(ty)
    # Plot the t-SNE visualization
    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(111)
    # Class names
    classes = ["CM", "CMM", "P1", "P2", "P3", "P31M", "P2M1", "P4", "P4G", "P4M", "P6", "P6M", "PG", "PGG", "PM", "PMG", "PMM"]
    names = {}
    for i in range(len(classes)):
        names[i] = classes[i]
    # color-wise coding to differentiate the classes
    colors = {label: np.random.randint(0, 255, 3) for label in names}
    for label in colors:
        idx = [i for i, l in enumerate(labels) if l == label]
        present_tx = np.take(tx, indices = idx)
        present_ty = np.take(ty, indices = idx)
        color = np.array(colors[label], dtype = np.float64) / 255
        ax.scatter(present_tx, present_ty, color = color, label = names[label])
    ax.legend(loc = 'best')
    # Save the plot
    plt.title("TSNE Visualization of Wallpaper Dataset")
    plt.savefig(filename, bbox_inches='tight')
    
def wallpaper_main(args):
    """
    Main function for training and testing the wallpaper classifier.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_classes = 17
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Load the wallpaper dataset
    data_root = os.path.join(args.data_root, 'Wallpaper')
    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set)):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set))

    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Data augmentation for training: 34000 images
    # Random Crop, Random Horizontal Flip, Random Vertical Flip, Resize, Grayscale, ToTensor, Normalize
    if args.aug_train:
        transform_1 = transforms.Compose([
            transforms.RandomCrop((150, 150)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        transform_2 = transforms.Compose([
            transforms.RandomRotation(360),
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])

    # If data augmentation is used, concatenate the augmented and normal datasets to get 34000 training images
    if args.aug_train:
        train_dataset_aug_1 = ImageFolder(os.path.join(data_root, 'train'), transform=transform_1)
        train_dataset_aug_2 = ImageFolder(os.path.join(data_root, 'train'), transform=transform_2)        
        train_dataset_normal = ImageFolder(os.path.join(data_root, 'train'), transform=transform)
        train_dataset = ConcatDataset([train_dataset_aug_1, train_dataset_aug_2, train_dataset_normal])
    else:
        train_dataset = ImageFolder(os.path.join(data_root, 'train'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = ImageFolder(os.path.join(data_root, args.test_set), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    print(f"Training on {len(train_dataset)} images, testing on {len(test_dataset)} images.")

    # Initialize the improved model if --improved arg is passed
    if args.improved:
        # If data augumentation is used, use the VGG16 model
        if args.aug_train:
            model = VGG16(num_classes = 17).to(device)
        # If data augmentation is not used, use the CNN2 model
        else:
            model = CNN2(input_channels = 1, img_size = args.img_size, num_classes = num_classes).to(device)
    # Initialize the baseline model
    else:
        model = CNN(input_channels = 1, img_size = args.img_size, num_classes = num_classes).to(device)

    # Initialize the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train + test the model
    model, per_epoch_loss, per_epoch_acc, train_preds, train_targets = train(model, train_loader, optimizer, criterion, args.num_epochs, 
                                                                             args.log_interval, device )
    test_loss, test_acc, test_preds, test_targets = test(model, test_loader, device, criterion)

    # Get stats 
    classes_train, overall_train_mat = get_stats(train_preds, train_targets, num_classes)
    classes_test, overall_test_mat = get_stats(test_preds, test_targets, num_classes)


    print(f'\n\nTrain accuracy: {per_epoch_acc[-1]*100:.3f}')
    print(f'Test accuracy: {test_acc*100:.3f}')

    if not os.path.exists(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats')):
        os.makedirs(os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats'))
    overall_file_name = os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'stats', 'overall.npz')

    # Visualize the a specific convolutional layer
    overall_layer_name = os.path.join(args.save_dir, 'Wallpaper', args.test_set, f'layer_{args.layer}.png')
    visualize_layer(model, test_loader, args.layer, args.device, overall_layer_name)

    # Visualize the t-SNE representation of the test set
    overall_tsne_test = os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'tsne_test.png')
    print("Visualizing t-SNE representation of Test Set...")
    visualize_tsne(model, test_loader, args.device, overall_tsne_test)
    # Visualize the t-SNE representation of the training set
    overall_tsne_train = os.path.join(args.save_dir, 'Wallpaper', args.test_set, 'tsne_train.png')
    print("Visualizing t-SNE representation of Train Set...")
    visualize_tsne(model, train_loader, args.device, overall_tsne_train)

    # Print the standard deviation of the accuracy over the 17 classes
    print(f"Training Accuracy Standard Deviation over 17 Classes: {np.std(classes_train):.5f}")
    print(f"Test Accuracy Standard Deviation over 17 Classes: {np.std(classes_test):.5f}")

    np.savez(overall_file_name, classes_train=classes_train, overall_train_mat=overall_train_mat, 
                classes_test=classes_test, overall_test_mat=overall_test_mat,
                per_epoch_loss=per_epoch_loss, per_epoch_acc=per_epoch_acc, 
                test_loss=test_loss, test_acc=test_acc)

def taiji_main(args):
    """
    Main function for training and testing the taiji classifier.
    Args:
        args (argparse.Namespace): Arguments.
    """
    num_subs = args.num_subs
    num_forms = 46 # Number of taiji forms, hardcoded :p
    sub_train_acc = np.zeros(num_subs)
    sub_class_train = np.zeros((num_subs, num_forms))
    sub_test_acc = np.zeros(num_subs)
    sub_class_test = np.zeros((num_subs, num_forms))
    overall_train_mat = np.zeros((num_forms, 1))
    overall_test_mat = np.zeros((num_forms, 1))

    if not os.path.exists(os.path.join(args.save_dir, 'Taiji', args.fp_size)):
        os.makedirs(os.path.join(args.save_dir, 'Taiji', args.fp_size))

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    # Seed torch and numpy
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    # For each subject LOSO
    for i in range(num_subs):
        print('\n\nTraining subject: {}'.format(i+1))

        train_data = TaijiData(data_dir='data', subject=i+1, split='train', fp_size = args.fp_size)
        test_data = TaijiData(data_dir ='data', subject=i+1, split='test', fp_size = args.fp_size)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        # Initialize the baseline model
        if args.baseline:
            model = MLP(input_dim = train_data.data_dim, hidden_dim = 1024, output_dim = num_forms).to(device)
        # Initialize the improved model
        if args.improved:
            if args.fp_size == 'lod4':
                hidden_dim_1, hidden_dim_2 = 512, 256
            else:
                hidden_dim_1, hidden_dim_2 = 1024, 1024
            model = MLP2(input_dim = train_data.data_dim, hidden_dim_1 = hidden_dim_1, hidden_dim_2 = hidden_dim_2, output_dim = num_forms).to(device)

        # Initialize the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Train + test the model
        model, train_losses, per_epoch_train_acc, train_preds, train_targets \
                        = train(model, train_loader, optimizer, criterion, args.num_epochs, args.log_interval, device)
        test_loss, test_acc, test_pred, test_targets = test(model, test_loader, device, criterion)

        # Print accs to three decimal places
        sub_train_acc[i] = per_epoch_train_acc[-1]
        sub_test_acc[i] = test_acc
        print(f'Subject {i+1} Train Accuracy: {per_epoch_train_acc[-1]*100:.3f}')
        print(f'Subject {i+1} Test Accuracy: {test_acc*100:.3f}')
        
        # Save all stats (you can save the model if you choose to)
        if not os.path.exists(os.path.join(args.save_dir, 'Taiji', args.fp_size, 'stats')):
            os.makedirs(os.path.join(args.save_dir, 'Taiji', args.fp_size, 'stats'))

        sub_file = os.path.join(args.save_dir, 'Taiji',args.fp_size, 'stats', 'sub_{}.npz'.format(i+1))
        classes_train, conf_mat_train = get_stats(train_preds, train_targets, num_forms)
        classes_test, conf_mat_test = get_stats(test_pred, test_targets, num_forms)
        sub_class_train[i, :] = classes_train
        sub_class_test[i, :] = classes_test
        overall_train_mat = overall_train_mat + (1/num_subs) * conf_mat_train 
        overall_test_mat = overall_test_mat + (1/num_subs) * conf_mat_test 
        np.savez(sub_file, train_losses=train_losses, per_epoch_acc=per_epoch_train_acc, test_acc=test_acc,
                 conf_mat_train=conf_mat_train, conf_mat_test=conf_mat_test)

    # Save overall stats
    overall_train_acc = np.mean(sub_train_acc)
    overall_test_acc = np.mean(sub_test_acc)
    print(f"\n\nOverall Train Accuracy: {overall_train_acc:.3f}")
    print(f"Overall Test Accuracy: {overall_test_acc:.3f}")

    # Print standard deviations over subjects
    print(f"Standard Deviation over subjects for Training Set: {np.std(sub_train_acc):.5f}")
    print(f"Standard Deviation over subjects for Test Set: {np.std(sub_test_acc):.5f}")

    overall_file_name = os.path.join(args.save_dir, 'Taiji', args.fp_size, 'stats', 'overall.npz')
    np.savez(overall_file_name, sub_train_acc = sub_train_acc, sub_class_train=sub_class_train,
             sub_test_acc=sub_test_acc, sub_class_test=sub_class_test, overall_train_mat=overall_train_mat, overall_test_mat=overall_test_mat)


if __name__ == '__main__':
    args = arg_parse()

    if args.dataset == 'Wallpaper':
        if args.train:
            wallpaper_main(args)
        visualize(args, dataset='Wallpaper')
        plot_training_curve(args)
    else: 
        if args.train:
            taiji_main(args)
        visualize(args, dataset='Taiji')
        plot_training_curve(args)


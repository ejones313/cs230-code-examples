"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from tqdm import trange

import utils
import model.net as net
from model.data_loader import DataLoader
from evaluate import evaluate
import data_prep
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def unscramble(output, lengths, original_indices, batch_size):
    a = (torch.from_numpy(np.array(lengths) - 1)).view(-1,1).expand(output.size(0),output.size(2))
    final_ids = (Variable(torch.from_numpy(np.array(lengths) - 1))).view(-1,1).expand(output.size(0),output.size(2)).unsqueeze(1)
    #Expand is incorrect - wrong axis?
    print(final_ids.data)
    print(output.shape)
    final_outputs = output.gather(1, final_ids).squeeze()
    #final_outputs = torch.gather(output, 1, final_ids).squeeze()

    mapping = original_indices.view(-1,1).expand(batch_size, output.size(1))
    unscrambled_outputs = final_outputs.gather(0, Variable(mapping))

    return unscrambled_outputs

# Does gradient descent on one epoch
def train(word_model, vid_model, word_optimizer, vid_optimizer, loss_fn, dataSet, metrics, params, anchor_is_phrase = True):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    word_model.train()
    vid_model.train()
    word_model.zero_grad()
    vid_model.zero_grad()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    

    batch_size = params.batch_size
    dataset_size = len(dataSet)
    num_batches = dataset_size // batch_size

    # Use tqdm for progress bar
    #for batch in dataLoader:
    for batch_num in range(0,num_batches-1):
        batch, indices = dataSet.get_batch(batch_size)

        anchor_batch = batch[0]
        positive_batch = batch[1]
        negative_batch = batch[2]

        anchor_indices = indices[0]
        positive_indices = indices[1]
        negative_indices = indices[2]

        # compute model output and loss
        if anchor_is_phrase:
            anchor_output = word_model(anchor_batch)
            positive_output = vid_model(positive_batch)
            negative_output = vid_model(negative_batch)
        else:
            anchor_output = vid_model(anchor_batch)
            positive_output = word_model(positive_batch)
            negative_output = word_model(negative_batch)

        anchor_output, anchor_lengths = nn.utils.rnn.pad_packed_sequence(anchor_output)
        positive_output, positive_lengths = nn.utils.rnn.pad_packed_sequence(positive_output)
        negative_output, negative_lengths = nn.utils.rnn.pad_packed_sequence(negative_output)

        anchor_unscrambled = unscramble(anchor_output, anchor_lengths, anchor_indices, batch_size)
        positive_unscrambled = unscramble(positive_output, positive_lengths, positive_indices, batch_size)
        negative_unscrambled = unscramble(negative_output, negative_lengths, negative_indices, batch_size)

        #print(type(anchor_output))

        '''anchor_unscrambled = torch.autograd.Variable(torch.zeros(anchor_output.shape))
        positive_unscrambled = torch.autograd.Variable(torch.zeros(positive_output.shape))
        negative_unscrambled = torch.autograd.Variable(torch.zeros(negative_output.shape))
        #anchor_unscrambled = torch.zeros(anchor_output.shape)
        #positive_unscrambled = torch.zeros(positive_output.shape)
        #negative_unscrambled = torch.zeros(negative_output.shape)

        for i in range(batch_size):
            anchor_unscrambled[:,anchor_indices[i],:] = anchor_output[:,i,:].data
            positive_unscrambled[:,positive_indices[i],:] = positive_output[:,i,:].data
            negative_unscrambled[:,negative_indices[i],:] = negative_output[:,i,:].data

        #print(type(anchor_unscrambled))

        anchor_output = anchor_unscrambled
        positive_output = positive_unscrambled
        negative_output = negative_unscrambled'''

        print(type(anchor_output))

        #print(type(anchor_output))

        '''idx = (seq_sizes - 1).view(-1, 1).expand(output.size(0), output.size(2)).unsqueeze(1)
        decoded = output.gather(1, idx).squeeze()

        decoded[original_index] = decoded'''
        loss = 0
        for i in range(batch_size):
            print(type(loss_fn(anchor_output[-2:-1,i,:], positive_output[-2:-1,i,:], negative_output[-2:-1,i,:])))
            loss += loss_fn(anchor_output[-2:-1,i,:], positive_output[-2:-1,i,:], negative_output[-2:-1,i,:])
        loss = loss/batch_size

        print(loss)

        # clear previous gradients, compute gradients of all variables wrt loss
        vid_optimizer.zero_grad()
        word_optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        word_optimizer.step()
        vid_optimizer.step()

        # Evaluate summaries only once in a while
        if False:
        #if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.data[0]
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss.data[0])
        #t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    # compute mean of all metrics in summary
    #metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    #metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    #logging.info("- Train metrics: " + metrics_string)
    

def train_and_evaluate(phrase_model, vid_model, train_filename, val_filename, phrase_optimizer, vid_optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    #if restore_file is not None:
    #    restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
    #    logging.info("Restoring parameters from {}".format(restore_path))
    #    utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_acc = 0.0

    #Added
    train_dataset = data_prep.Dataset(train_filename)
    #train_loader = torch.utils.data.DataLoader(train_dataset, params.batch_size)
    """
    val_dataset = data_prep.Dataset(val_filename)
    val_loader = torch.utils.data.DataLoader(val_dataset)
    """
    #for epoch in range(params.num_epochs):
    for epoch in range(50):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        #TODO: Add params.filename

        train(phrase_model, vid_model, phrase_optimizer, vid_optimizer, loss_fn, train_dataset, metrics, params)
        train_dataset.reset_counter()
            
        # Evaluate for one epoch on validation set
        """
        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = data_loader.data_iterator(val_data, params, shuffle=False)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)
        
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()}, 
                               is_best=is_best,
                               checkpoint=model_dir)
            
        # If best_eval, best_save_path        
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        """
    

if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.word_embedding_dim = 300
    params.word_hidden_dim = 600
    params.vid_embedding_dim = 500
    params.vid_hidden_dim = 600

    params.batch_size = 1

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Set the logger

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    
    # load data
    """
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']
    """
    train_filename = 'subset_two.pkl'
    val_filename = 'foo.py'

    # specify the train and val dataset sizes
    """
    params.train_size = train_data['size']
    params.val_size = val_data['size']
    """

    logging.info("- done.")

    # Define the model and optimizer
    phrase_model = net.Net(params, True).cuda() if params.cuda else net.Net(params, True)
    vid_model = net.Net(params, False).cuda() if params.cuda else net.Net(params, False)
    phrase_optimizer = optim.Adam(phrase_model.parameters(), lr=params.learning_rate)
    vid_optimizer = optim.Adam(vid_model.parameters(), lr=params.learning_rate)
    
    # fetch loss function and metrics
    loss_fn = torch.nn.modules.loss.TripletMarginLoss()
    #metrics = net.metrics
    metrics = None

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(phrase_model, vid_model, train_filename, val_filename, phrase_optimizer, vid_optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)

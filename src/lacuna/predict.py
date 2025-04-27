import os
import numpy as np

from torch.autograd import Variable
import torch.nn.functional as F
from IPython.core.debugger import set_trace

import torch
from .utils import make_reproducible

def predict(pred_data, model, buffer, gpu, num_classes, shrink_pixel):
    """
    Generates predictions from a trained model for each tile in a given dataset.
    
    This function processes each batch of data from the DataLoader, applies the 
    model for prediction, and then compiles the results into a list of numpy arrays 
    representing the scores for each class.
    
    Args:
        pred_data (tuple): Contains DataLoader, metadata, tile information, and year. 
                           DataLoader batches the data for prediction.
        model (torch.nn.Module): The trained model used for prediction.
        buffer (int): The size of the buffer to be removed from the border of each 
                      prediction tile.
        gpu (bool): If True, use GPU for prediction; otherwise, use CPU.
        num_classes (int): number of semantic categories in the output prediction.
        shrink_pixel (int): Number of pixels to trim from the edges of each 
                            output canvas. This is used to adjust the final size 
                            of the prediction output, accounting for the buffer 
                            and any edge effects in prediction processing.
    
    Returns:
        A list of numpy arrays, each representing the prediction score for a class across 
        the entire dataset.
    
    """
    pred_data, meta, tile, year = pred_data
    meta.update({
        'dtype': 'int8'
    })

    model.eval()

    # create dummy tile
    canvas_score_ls = [None] * (num_classes - 1)

    for img, index_batch, _, _ in pred_data:

        img = Variable(img, requires_grad=False)

        # GPU setting
        if gpu:
            img = img.cuda()
        
        with torch.no_grad():
            out = F.softmax(model(img), 1)
            torch.cuda.empty_cache()

        batch, nclass, height, width = out.size()
        chip_height = height - buffer * 2
        chip_width = width - buffer * 2
        max_index_0 = meta['height'] - chip_height
        max_index_1 = meta['width'] - chip_width

        # new by taking average
        for i in range(batch):
            index = (index_batch[0][i], index_batch[1][i])
            # only score here
            for n in range(nclass - 1):
                out_score = out[
                    :, 
                    n + 1, 
                    (index[0] != 0) * buffer : (index[0] != 0) * buffer + chip_height + (index[0]==0 or index[0]==max_index_0) * buffer,
                    (index[1] != 0) * buffer:(index[1] != 0) * buffer + chip_height + (index[1]==0 or index[1]==max_index_1) * buffer
                ].data[i].cpu().numpy() * 100
                
                contains_nan = np.isnan(out_score).any()
                print('contains_nan:', contains_nan)

                out_score = out_score.astype(meta['dtype'])
                score_height, score_width = out_score.shape

                if canvas_score_ls[n] is None:
                    canvas_score = np.zeros(
                        (meta['height'] + buffer * 2, 
                         meta['width'] + buffer * 2),
                        dtype=meta['dtype'])

                    canvas_score[
                        index[0] + buffer * (index[0] != 0) : index[0] + buffer * (index[0] != 0) + score_height,
                        index[1] + buffer * (index[1] != 0) : index[1] + buffer * (index[1] != 0) + score_width
                    ] = out_score

                    canvas_score_ls[n] = canvas_score

                else:
                    # Update existing canvas_score
                    canvas_score_ls[n][
                        index[0] + buffer * (index[0] != 0) : index[0] + buffer * (index[0] != 0) + score_height,
                        index[1] + buffer * (index[1] != 0) : index[1] + buffer * (index[1] != 0) + score_width
                    ] = out_score

    for j in range(len(canvas_score_ls)):
        canvas_score_ls[j] = canvas_score_ls[j][
            shrink_pixel:meta['height'] + buffer * 2 - shrink_pixel, 
            shrink_pixel:meta['width'] + buffer * 2 - shrink_pixel
        ]

    return canvas_score_ls


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def mc_predict(pred_data, model, num_mc_trials, buffer, gpu, num_classes, shrink_pixel):
    """
    Performs Monte Carlo (MC) predictions for each tile in a given dataset
    using a trained model.
    
    This function applies the model multiple times with MC dropout enabled, 
    collecting predictions from each trial. It compiles the results into a 
    list where each element represents the stacked prediction scores for each 
    class across all MC trials.
    
    Args:
        pred_data (tuple): Contains DataLoader, metadata, tile information, and year. 
                           DataLoader batches the data for prediction.
        model (torch.nn.Module): The trained model used for prediction.
        num_mc_trials (int): The number of Monte Carlo trials to perform.
        buffer (int): The size of the buffer to be removed from the border of each 
                      prediction tile.
        gpu (bool): If True, use GPU for prediction; otherwise, use CPU.
        num_classes (int): number of semantic categories in the output prediction.
        shrink_pixel (int): Number of pixels to trim from the edges of each 
                            output canvas. This is used to adjust the final size 
                            of the prediction output, accounting for the buffer 
                            and any edge effects in prediction processing.
    """
    #set_trace()
    pred_data, meta, tile, year = pred_data
    
    #print(len(pred_data))
    
    meta.update({
        'dtype': 'int8'
    })

    mc_canvas_score_ls = []
    #canvas_score_ls = [None] * (num_classes - 1)
    mc_score_dict = {}


    for mc_trial in range(num_mc_trials):
        print('trail No. {}'.format(mc_trial))
        make_reproducible(seed=mc_trial)
        model.eval()
        enable_dropout(model)
        canvas_score_ls = []

        for img, index_batch, _, _ in pred_data:

            img = Variable(img, requires_grad=False)
            if gpu:
                img = img.cuda()

            #out = F.softmax(model(img), 1)
            with torch.no_grad():
                out = F.softmax(model(img), 1)
                torch.cuda.empty_cache()

            batch, nclass, height, width = out.size()
            assert nclass == num_classes
            chip_height = height - buffer * 2
            chip_width = width - buffer * 2
            max_index_0 = meta['height'] - chip_height
            max_index_1 = meta['width'] - chip_width
            
            #print(f"max_index_0: {max_index_0}")
            #print(f"max_index_1: {max_index_1}")

            for i in range(batch):
                index = (index_batch[0][i], index_batch[1][i])

                for n in range(nclass - 1):
                    out_score = out[:, n + 1,
                                (index[0] != 0) * buffer: (index[0] != 0) * buffer + chip_height + (
                                            index[0] == 0 or index[0] == max_index_0) * buffer,
                                (index[1] != 0) * buffer: (index[1] != 0) * buffer + chip_height + (
                                            index[1] == 0 or index[1] == max_index_1)
                                                          * buffer].data[i].cpu().numpy() * 100
                    out_score = out_score.astype(meta['dtype'])
                    # out_score = np.expand_dims(out_score, axis=0)
                    score_height, score_width = out_score.shape

                    try:
                        # if canvas_score_ls[n] exists
                        canvas_score_ls[n][
                        index[0] + buffer * (index[0] != 0): index[0] + buffer * (index[0] != 0) + score_height,
                        index[1] + buffer * (index[1] != 0): index[1] + buffer * (
                                    index[1] != 0) + score_width] = out_score
                    except:
                        # create masked canvas_score_ls[n]
                        canvas_score = np.zeros((meta['height'] + buffer * 2, meta['width'] + buffer * 2),
                                                dtype=meta['dtype'])

                        canvas_score[
                        index[0] + buffer * (index[0] != 0): index[0] + buffer * (index[0] != 0) + score_height,
                        index[1] + buffer * (index[1] != 0): index[1] + buffer * (
                                    index[1] != 0) + score_width] = out_score
                        canvas_score_ls.append(canvas_score)

        for j in range(len(canvas_score_ls)):
            canvas_score_ls[j] = canvas_score_ls[j][shrink_pixel:meta['height'] + buffer * 2 - shrink_pixel,
                                 shrink_pixel:meta['width'] + buffer * 2 - shrink_pixel]

        mc_score_dict[mc_trial] = canvas_score_ls

    for i in range(len(list(mc_score_dict.values())[0])):
        mc_canvas_score_ls.append(
            np.concatenate([np.expand_dims(value_ls[i], 0) for value_ls in mc_score_dict.values()], 0))



    #print('stop after the first prediction completed')
    #print(mc_canvas_score_ls[0])
    #print(len(mc_canvas_score_ls[0]))
    #print(mc_score_dict[0][1])
    #print('-------')
    #print(mc_score_dict[1][1])
    # print(np.array_equal(mc_canvas_score_ls[0][0], 
    #                     mc_canvas_score_ls[0][1]))

    #exit()


    del mc_score_dict
    return mc_canvas_score_ls

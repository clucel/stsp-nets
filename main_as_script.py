from networks import DMTSNet #stsp and fixed (same but no x+u) networks
from spatial_task import DMTSDataModule #spatial version, distraction code removed

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
from pytorch_lightning import Trainer

def import_model_task_trainer(
    logger, 
    label = '',
    
    rnn_type = 'stsp', 
    nonlinearity = 'relu', 
    hidden_size = 100, 
    learning_rate = .02, 
    gamma = .005, 
    act_reg = 1e-3, 
    param_reg = 1e-4, 
    init_method = 'log',
    
    noise_level = 0.01,
    difficulty_level = 2,
    
    epochs = 20, 
    optimizer = "Adam",
    BATCH_SIZE = 256
):

    '''
    Function to load model and trainer
    INPUTS (with defaults)
    logging:
        logger, 
        label = '',
    network:
        rnn_type = 'stsp', 
        nonlinearity = 'relu', 
        hidden_size = 100, 
        learning_rate = .02, 
        gamma = .005, 
        act_reg = 1e-3, 
        param_reg = 1e-4, 
        init_method = 'log',
    task:
        noise_level = 0.01,
        difficulty_level = 2,
    training:
        epochs = 20, 
        BATCH_SIZE = 256,
        optimizer = "Adam"

    Returns model, task, trainer
    '''

    #CONSTANT VARIABLES:
    input_size = 3  #2 ports + 1 fixation (center)
    output_size = 3  #same as inputs
    dt_ann = 15
    alpha = dt_ann / 100
    alpha_W = dt_ann / 100
    g = 0.9

    AVAIL_GPUS = torch.cuda.device_count()
    # BATCH_SIZE = 256 if AVAIL_GPUS else 32
    print(f"{AVAIL_GPUS} GPU availible")
    print(f"The batch size is set to {BATCH_SIZE}")


    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        dirpath="_lightning_sandbox/checkpoints/",
        filename="rnn-sample-DMTS-{epoch:02d}-{val_acc:.2f}--"
        + rnn_type
        + "--"
        + nonlinearity,
        every_n_epochs=epochs,
        mode="max",
        save_last=True,
    )

    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        label + f"-{rnn_type}-{nonlinearity}-{hidden_size}-{gamma}-{learning_rate}-{act_reg}-{param_reg}-{init_method}-{noise_level}-{difficulty_level}-{optimizer}-{BATCH_SIZE}-" + "{epoch:02d}-{val_acc:.2f}"
    )
    #f"rnn={rnn_type}--nl={nonlinearity}--hs={hidden_size}--lr={learning_rate}--act_reg={act_reg}--gamma={gamma}--param_reg={param_reg}--" + "{epoch:02d}--{val_acc:.2f}"

    early_stop_callback = EarlyStopping(monitor="val_acc", stopping_threshold=0.8, mode="max", patience=50)

    model = DMTSNet(
        rnn_type,
        input_size,
        hidden_size,
        output_size,
        dt_ann,
        alpha,
        alpha_W,
        gamma,
        nonlinearity,
        learning_rate,
        init_method,
        optimizer,
        difficulty_level
    )

    model.act_reg = act_reg
    model.param_reg = param_reg
    
    #load task
    task = DMTSDataModule(dt_ann=dt_ann, noise_level=noise_level, difficulty_level=difficulty_level)
    
    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="auto",
        log_every_n_steps=10,
        logger=logger #this is new
    )
    
    return model, task, trainer
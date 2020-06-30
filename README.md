# Deep Collaborative Filtering
In this project a deep neural network is used to create a collaborative filtering
system for movies recommendation.  

## Dataset

## Network architecture

## Experimentation
![Train loss](docs/train_loss.png)

![Evak loss](docs/eval_loss.png)

**Note**: Additional information on the training process and hyperparameters can be found [link](https://app.wandb.ai/polmonroig/mrs?workspace=user-polmonroig "here").

## Further research
After evaluating the model some issues where encountered where a lot of users where
assigned similar movies, this might be due to the distribution of the dataset itself as
it might be biased to specific movies or the nature of the model.

## Requierements and Usage  
To use the project the following requirements are needed, some are just needed for training
such as torch and pandas for the data analysis.
`
pandas~=1.0.4
numpy~=1.18.5
scikit-learn~=0.23.1
torch~=1.5.1
scipy~=1.4.1
`
### Training
As for the training the following parameters must be used
`
usage: train.py [-h] [--denoiser] [--basic] [--tied_weights] [--not_tied] [--n_epochs N_EPOCHS] [--batch_size BATCH_SIZE]
                [--verbose_epochs VERBOSE_EPOCHS] [--lr LR]

optional arguments:
  -h, --help            show this help message and exit
  --denoiser            Use denoiser autoencoder
  --basic               Use basic autoencoder
  --tied_weights        Tie encoder/decoder weights
  --not_tied            Use separate encoder/decoder weights
  --n_epochs N_EPOCHS   Number of epochs for the training loop
  --batch_size BATCH_SIZE
                        Batch size of the training and eval set
  --verbose_epochs VERBOSE_EPOCHS
                        Number of epochs per training verbose output
  --lr LR               Learning rate of the optimizer
`
### Evaluation
The evaluation script is separated with two functionlities, first, the model must
be converted to ONNX format, this enables a faster inference and better compatibility.
After the ONNX model is generated the inference can be done on a scpy sparse matrix.
`
usage: eval.py [-h] [--evaluate] [--generate] [--model MODEL] [--batch_size BATCH_SIZE] [--data DATA]

optional arguments:
  -h, --help            show this help message and exit
  --evaluate            Evaluate a specified model
  --generate            Convert a pytorch model to ONNX
  --model MODEL         Model to convert or evaluate
  --batch_size BATCH_SIZE
                        Batch size of the input data
  --data DATA           Data to evaluate
`

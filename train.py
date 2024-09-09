from config import config

from src.trainer import Trainer
from src.data_process import load_train_dataframe

cfg = config['train']

def main():

    data = load_train_dataframe()
    trainer = Trainer(data)
    trainer.train_stacked_model()

if __name__ == '__main__':
    main()
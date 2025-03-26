# main.py
from train import train_and_evaluate

if __name__ == "__main__":
    # Define paths
    folder_path = "/home/featurize/work/ylx/MEA/data/calcium_data_all/GABA_all/day45_GABA"  # Replace with your folder
    model_save_path = "/home/featurize/work/ylx/MEA/gae/xgboost_model.json"

    # Train and evaluate the model
    train_and_evaluate(folder_path, model_save_path)
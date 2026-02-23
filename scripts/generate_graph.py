import torch
import matplotlib.pyplot as plt
import argparse
from config.config import default_config as config

def visualize_result(model_path: str,device):
    checkpoint = torch.load(model_path, map_location=torch.device(device))

    losses = checkpoint['losses']
    steps = checkpoint['steps']
    plt.plot(range(steps),losses)
    
    plt.xlabel("Steps")
    plt.ylabel("Losses")
    plt.show()



def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the results of the training process.")
    parser.add_argument('--model_path', type=str, help='Path to the saved model checkpoint.')

    args = parser.parse_args()

    graph = visualize_result(args.model_path,
                             config['device'])


if __name__ == "__main__":
    main()
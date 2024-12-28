import matplotlib.pyplot as plt
import pandas as pd

def plot_results(train_hist, valid_hist, acc_train_hist, acc_valid_hist, model_name, neurons, activation, save_path):
    epochs = len(train_hist)
    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_hist, label='Train Loss')
    plt.plot(range(epochs), valid_hist, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} | Neurons: {neurons} | Activation: {activation} (Loss)')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), acc_train_hist, label='Train Accuracy')
    plt.plot(range(epochs), acc_valid_hist, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} | Neurons: {neurons} | Activation: {activation} (Accuracy)')
    plt.legend()

    # Save and Show Plot
    plt.tight_layout()
    plt.savefig(f'{save_path}/{model_name}_results.png')
    plt.show()

# Save results to CSV
def save_results_to_csv(results, file_name):
    df = pd.DataFrame(results)
    df.to_csv(file_name, index=False)
    print(f'Results saved to {file_name}')


import csv
import matplotlib.pyplot as plt

# CSV file path
csv_path = 'logs/training_log.csv'

# Lists to store epoch, loss, val_acc
epochs = []
train_loss = []
val_acc = []

# Read CSV
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row['epoch']))
        train_loss.append(float(row['train_loss']))
        val_acc.append(float(row['val_acc']))

# Plot training loss and validation accuracy
plt.figure(figsize=(10,4))

# Training loss subplot
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, 'o-', color='blue', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.legend()

# Validation accuracy subplot
plt.subplot(1,2,2)
plt.plot(epochs, val_acc, 'o-', color='green', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save the figure as a PNG file
plt.savefig('logs/training_curves.png')  # <-- 保存路径可自定义
plt.show()

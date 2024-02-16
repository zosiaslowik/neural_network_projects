import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

input_data = []
with open('input_data.csv') as file:
    for line in file:
        line_split = line.split(', ')
        data = []
        for elem in line_split:
            data.append(float(elem))
        input_data.append(data)
input_data = np.array(input_data)
target_data = []
with open('target_data.csv') as file:
    for line in file:
        line_split = line.split(', ')
        data = []
        for i in range(2):
            data.append(float(line_split[i]))
        target_data.append(data)
input_data=np.array(input_data)
target_data=np.array(target_data)

# Standaryzacja danych
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Struktura sieci neuronowej
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)  # Warstwa wyjściowa z dwoma neuronami odpowiadającymi za pozycję (x, y)
])

# Algorytm uczący
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# K-krotna walidacja krzyżowa
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fig, axes = plt.subplots(num_folds, 2, figsize=(12, 2 * num_folds))
fig.subplots_adjust(hspace=0.5)


for fold, (train_idx, val_idx) in enumerate(kfold.split(input_data, target_data)):
    X_train, X_val = input_data[train_idx], input_data[val_idx]
    y_train, y_val = target_data[train_idx], target_data[val_idx]

    # Trenowanie modelu
    history = model.fit(np.array(X_train), np.array(y_train), epochs=50, batch_size=32,
                        validation_data=(np.array(X_val), np.array(y_val)))

    # Testowanie modelu na zestawie walidacyjnym
    scores = model.evaluate(np.array(X_val), np.array(y_val), verbose=0)
    print(f"Fold {fold + 1}/{num_folds} - Loss: {scores}")

    # Training history
    axes[fold, 0].plot(history.history['loss'], label='Training Loss')
    axes[fold, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[fold, 0].set_title(f'Fold {fold + 1} Training History')
    axes[fold, 0].set_xlabel('Epochs')
    axes[fold, 0].set_ylabel('Loss')
    axes[fold, 0].legend()

    # true vs predicted
    axes[fold, 1].scatter(y_val[:, 0], model.predict(X_val)[:, 0], label='True vs Predicted (X)')
    axes[fold, 1].scatter(y_val[:, 1], model.predict(X_val)[:, 1], label='True vs Predicted (Y)')
    axes[fold, 1].set_title(f'Fold {fold + 1} True vs Predicted Values')
    axes[fold, 1].set_xlabel('True Values')
    axes[fold, 1].set_ylabel('Predicted Values')
    axes[fold, 1].legend()


plt.tight_layout()
plt.show()

sample_indices = np.random.choice(len(X_val), size=25, replace=False)
sample_X = X_val[sample_indices]
sample_y_true = y_val[sample_indices]
sample_y_pred = model.predict(sample_X)


print("Sample Data Points vs Predicted Values:")
for i in range(len(sample_indices)):
    print(f"Data Point {i + 1} - True: {sample_y_true[i]}, Predicted: {sample_y_pred[i]}")

plt.figure(figsize=(8, 6))
plt.scatter(sample_y_true[:, 0], sample_y_true[:, 1], label='True')
plt.scatter(sample_y_pred[:, 0], sample_y_pred[:, 1], label='Predicted')
plt.title('Sample True vs Predicted Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
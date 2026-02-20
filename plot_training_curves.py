import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to results.json
data_path = os.path.join('outputs', 'results.json')

# Read all lines and parse JSON objects
with open(data_path, 'r') as f:
    lines = f.readlines()

# Flatten all JSON objects (some lines are lists, some are dicts)
results = []
for line in lines:
    try:
        obj = json.loads(line)
        if isinstance(obj, list):
            results.extend(obj)
        else:
            results.append(obj)
    except Exception:
        continue

# Filter only global round results (exclude device/client evals, etc.)
global_results = [r for r in results if isinstance(
    r, dict) and r.get('type') == 'global' and 'round' in r]
global_results = sorted(global_results, key=lambda x: x['round'])

rounds = [r['round'] for r in global_results]
loss = [r['loss'] for r in global_results]
accuracy = [r['accuracy'] for r in global_results]
f1_macro = [r['f1_macro'] for r in global_results]
auc_macro = [r['auc_macro'] for r in global_results]

# Per-class metrics (as lists of lists)
per_class_f1 = np.array([r['per_class_f1'] for r in global_results])
per_class_precision = np.array(
    [r['per_class_precision'] for r in global_results])
per_class_recall = np.array([r['per_class_recall'] for r in global_results])

# Plot Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(rounds, loss, marker='o')
plt.title('Loss Curve')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('outputs/loss_curve.png')
plt.close()

# Plot Accuracy Curve
plt.figure(figsize=(8, 5))
plt.plot(rounds, accuracy, marker='o', color='green')
plt.title('Accuracy Curve')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig('outputs/accuracy_curve.png')
plt.close()

# Plot F1 Macro Curve
plt.figure(figsize=(8, 5))
plt.plot(rounds, f1_macro, marker='o', color='purple')
plt.title('F1 Macro Curve')
plt.xlabel('Round')
plt.ylabel('F1 Macro')
plt.grid(True)
plt.savefig('outputs/f1_macro_curve.png')
plt.close()

# Plot AUC Macro Curve
plt.figure(figsize=(8, 5))
plt.plot(rounds, auc_macro, marker='o', color='orange')
plt.title('AUC Macro Curve')
plt.xlabel('Round')
plt.ylabel('AUC Macro')
plt.grid(True)
plt.savefig('outputs/auc_macro_curve.png')
plt.close()

# Plot Per-Class F1 Curves
plt.figure(figsize=(10, 6))
for i in range(per_class_f1.shape[1]):
    plt.plot(rounds, per_class_f1[:, i], marker='o', label=f'Class {i} F1')
plt.title('Per-Class F1 Curves')
plt.xlabel('Round')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.savefig('outputs/per_class_f1_curves.png')
plt.close()

# Plot Per-Class Precision Curves
plt.figure(figsize=(10, 6))
for i in range(per_class_precision.shape[1]):
    plt.plot(rounds, per_class_precision[:, i],
             marker='o', label=f'Class {i} Precision')
plt.title('Per-Class Precision Curves')
plt.xlabel('Round')
plt.ylabel('Precision')
plt.legend()
plt.grid(True)
plt.savefig('outputs/per_class_precision_curves.png')
plt.close()

# Plot Per-Class Recall Curves
plt.figure(figsize=(10, 6))
for i in range(per_class_recall.shape[1]):
    plt.plot(rounds, per_class_recall[:, i],
             marker='o', label=f'Class {i} Recall')
plt.title('Per-Class Recall Curves')
plt.xlabel('Round')
plt.ylabel('Recall')
plt.legend()
plt.grid(True)
plt.savefig('outputs/per_class_recall_curves.png')
plt.close()

print('All metric curves have been generated and saved in the outputs/ directory.')

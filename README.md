# Mushroom-Edibility-Classification

## Model Results

Each model was trained and evaluated under three different configurations. The evaluation results are saved as ``.pt`` files in JSON format. Each result file contains the following information:

- **accuracy:** Final test set accuracy of the model.
- **model_preds:** Model predictions on the test set.
- **true_labels:** Ground-truth labels for the test set, provided in the same order as predictions to enable reproducible analysis.
- **train_accuracies:** Training accuracies recorded at each epoch.
- **train_losses:** Training losses recorded at each epoch.
- **val_accuracies:** Validation accuracies recorded at each epoch.
- **val_losses:** Validation losses recorded at each epoch.
- **experiment_time:** Total elapsed time for model training and evaluation.

from itertools import cycle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, label_binarize

class PlotHelper:


    def _layers_to_str(self, v):
        if isinstance(v, (list, tuple)):
            return '-'.join(map(str, v))
        return str(v)
        
    # Parses hyperparameter strings from a DataFrame into structured columns.
    # This is a private helper method for internal use.
    #
    # Args:
    #     df (pd.DataFrame): DataFrame with a 'hyperparameters' column.
    #
    # Returns:
    #     pd.DataFrame: The DataFrame with new columns for each hyperparameter.
    def _parse_hyperparameters(self, df):
    
        df = df.copy()
        hp = df['hyperparameters'].apply(ast.literal_eval)
        df['ACTIVATION'] = hp.apply(lambda d: d.get('ACTIVATION'))
        df['DROPOUT_RATE'] = hp.apply(lambda d: d.get('DROPOUT_RATE'))
        df['LEARNING_RATE'] = hp.apply(lambda d: d.get('LEARNING_RATE'))
        df['DENSE_LAYERS'] = hp.apply(lambda d: d.get('DENSE_LAYERS'))
        df['BATCH_SIZE'] = hp.apply(lambda d: d.get('BATCH_SIZE'))
        


        df['layers_str'] = df['DENSE_LAYERS'].apply(self._layers_to_str)
        return df


    # Safely parses a string representation of training history into a dictionary.
    # This is a private helper method for internal use.
    #
    # Args:
    #     history_str (str): The string representation of the history dictionary.
    #
    # Returns:
    #     dict: The parsed history dictionary, or an empty dictionary on failure.
    def _parse_history(self, history_str):
        if isinstance(history_str, dict):
            return history_str
        if isinstance(history_str, str):
            try:
                return ast.literal_eval(history_str)
            except (ValueError, SyntaxError):
                return {}
        return {}


    # Creates a lollipop plot to compare the validation loss of different model configurations.
    # This helps visualize which hyperparameter combinations performed best.
    #
    # Args:
    #     df (pd.DataFrame): DataFrame with model performance data.
    def plot_model_val_loss_score(self, df):
        df = self._parse_hyperparameters(df)

        group_cols = ['layers_str', 'LEARNING_RATE', 'DROPOUT_RATE', 'ACTIVATION']
        agg = df.groupby(group_cols, as_index=False).agg(val_loss=('val_loss', 'min'))

        agg['config_label'] = (
            'layers=' + agg['layers_str'].astype(str) +
            ' | lr=' + agg['LEARNING_RATE'].astype(str) +
            ' | dropout=' + agg['DROPOUT_RATE'].astype(str) +
            ' | act=' + agg['ACTIVATION'].astype(str)
        )

        order = agg.sort_values('val_loss', ascending=True)['config_label'].tolist()
        cat_to_pos = {cat: i for i, cat in enumerate(order)}
        agg['ypos'] = agg['config_label'].map(cat_to_pos)

        palette = {'relu': '#1f77b4', 'tanh': '#2ca02c', 'sigmoid': '#ff7f0e'}
        x_baseline = float(agg['val_loss'].max())

        fig_h = max(4, min(0.5 * max(3, len(order)), 20))
        fig, ax = plt.subplots(figsize=(11, fig_h))

        for act, color in palette.items():
            sub = agg[agg['ACTIVATION'] == act]
            ax.hlines(y=sub['ypos'], xmin=x_baseline, xmax=sub['val_loss'], color=color, alpha=0.35, linewidth=2)
            ax.scatter(sub['val_loss'], sub['ypos'], s=60, c=color, edgecolors='k', linewidths=0.4, alpha=0.95, label=str(act), zorder=3)

        ax.set_xscale('log')
        ax.invert_xaxis()
        ax.set_xlabel('Validation Loss (log scale; lower is better → right)')
        ax.set_ylabel('Configuration (layers | lr | dropout | act)')
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        ax.legend(title='Activation', loc='lower left', bbox_to_anchor=(1.02, 0), borderaxespad=0)
        plt.title("Validation Loss score by model configuration (log scale)")
        plt.tight_layout()
        plt.show()


    # Creates box plots to show the distribution of validation loss for different hyperparameter values.
    # This is useful for understanding the impact of each hyperparameter on model performance.
    #
    # Args:
    #     df (pd.DataFrame): DataFrame with model performance data.
    def plot_val_loss_by_parameter(self, df):
        df = self._parse_hyperparameters(df)
        
        params_to_plot = ['ACTIVATION', 'DENSE_LAYERS', 'DROPOUT_RATE', 'LEARNING_RATE', 'BATCH_SIZE']

        val_losses = pd.to_numeric(df['val_loss'], errors='coerce').dropna()
        if val_losses.empty:
            print("No validation loss data to plot.")
            return

        # Create a single figure with multiple subplots (one for each parameter)
        n = len(params_to_plot)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 5 * n))
        if n == 1:
            axes = [axes] # Make sure axes is always a list

        fig.suptitle('Validation Loss by Hyperparameter', fontsize=16, y=1.0)

        # Iterate through each hyperparameter and render its subplot
        for i, param_name in enumerate(params_to_plot):
            ax = axes[i]
            col_name = 'layers_str' if param_name == 'DENSE_LAYERS' else param_name
            
            # Filter out rows where the parameter is missing
            sub_df = df.dropna(subset=['val_loss', col_name]).copy()

            # as separate categories, which was the root cause of the plotting error.
            sub_df[col_name] = sub_df[col_name].astype(str)

            # Group the filtered data by the hyperparameter's values
            grouped = sub_df.groupby(col_name)['val_loss'].apply(list)
            
            # Sort the categories by their median loss for a clearer visualization
            order = grouped.apply(np.median).sort_values(ascending=False).index
            data_in_order = [grouped[k] for k in order]

            # Plot the boxplot
            ax.boxplot(data_in_order, vert=False, labels=order, patch_artist=True,
                       medianprops=dict(color='black', linewidth=1.2),
                       boxprops=dict(facecolor='#cfe2f3', edgecolor='#6c8ebf', linewidth=1.1))

            # Set titles and labels for the subplot
            ax.set_title(param_name, fontsize=14, pad=10)
            ax.set_ylabel(f'{param_name} option')
            ax.set_xlabel('Validation Loss (log scale; lower is better →)')
            ax.set_xscale('log')
            ax.invert_xaxis()
            ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=4.0) # Increase vertical padding
        plt.show()


    # Generic helper to plot a metric from training history over epochs.
    # This private method is used by public plotting functions for loss and accuracy.
    #
    # Args:
    #     history_str (str): String representation of the history dictionary.
    #     metric (str): The key for the metric in the history dict (e.g., 'val_loss').
    #     ylabel (str): The label for the y-axis.
    #     color (str): The color of the plot line.
    #     highlight_color (str): The color for the highlight marker.
    #     mode (str): 'max' to find the maximum value, 'min' to find the minimum.


    # Plots validation loss over epochs for the winning model configuration.
    #
    # Args:
    #     df_winner (pd.Series): A row from the performance DataFrame for the best model.
    def plot_val_loss_in_epochs(self, df_winner):
        history = self._parse_history(df_winner['history'])
        curve = history.get('val_loss')

        curve = [float(v) for v in curve]
        epochs = np.arange(1, len(curve) + 1)
        
        best_idx = int(np.argmin(curve))
        best_val = curve[best_idx]
        label = f"Min: {best_val:.3e} @ epoch {epochs[best_idx]}"
        title = 'Validation Loss across Epochs'

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, curve, marker='o', markersize=3, linewidth=1.5, color='tab:blue')
        
        ax.set_yscale('log')
        ylabel = "Validation Loss (log scale)"

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        ax.scatter(epochs[best_idx], best_val, color='red', s=50, zorder=5, label=label)
        ax.legend(loc='best')
        
        plt.tight_layout()
        plt.show()


    # Plots validation accuracy over epochs for the winning model configuration.
    #
    # Args:
    #     df_winner (pd.Series): A row from the performance DataFrame for the best model.
    def plot_val_accuracy_in_epochs(self, df_winner):
        history = self._parse_history(df_winner['history'])
        curve = history.get('val_accuracy')

        curve = [float(v) for v in curve]
        epochs = np.arange(1, len(curve) + 1)
        
        best_idx = int(np.argmax(curve))
        best_val = curve[best_idx]
        label = f"Max: {best_val:.4f} @ epoch {epochs[best_idx]}"
        title = 'Validation Accuracy across Epochs'

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, curve, marker='o', markersize=3, linewidth=1.5, color='tab:green')
        
        ylabel = "Validation Accuracy"

        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        ax.scatter(epochs[best_idx], best_val, color='blue', s=50, zorder=5, label=label)
        ax.legend(loc='lower right')
        
        if np.min(curve) > 0.9:
            ax.set_ylim(max(0.9, np.min(curve) - 0.01), min(1.0, np.max(curve) + 0.01))

        plt.tight_layout()
        plt.show()


    # Plots a confusion matrix heatmap to evaluate classification performance.
    #
    # Args:
    #     y_true (array-like): Ground truth labels (as integers).
    #     y_pred (array-like): Predicted labels from the model (as integers).
    def plot_heatmap(self, y_true, y_pred, classes):
        # Ensure labels are integers from 0 to n_classes-1
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))

        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix | Validation Accuracy: {accuracy:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()


    # Plots the Receiver Operating Characteristic (ROC) curve for each class.
    # This helps visualize the model's ability to distinguish between classes.
    #
    # Args:
    #     y_true (array-like): Ground truth labels (as integers).
    #     y_probs (array-like): Predicted probabilities for each class from the model.
    def plot_roc_curve(self, y_true, y_probs, classes):
        n_classes = len(classes)
        y_true_bin = label_binarize(y_true, classes=classes)

        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(9, 7))
        colors = cycle(sns.color_palette("husl", n_classes))
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2.5, alpha=0.7,
                     label=f'ROC class {classes[i]} (AUC = {roc_auc[i]:.2f})')

        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


    # Generates and plots the PCA of the validation data colored by predicted labels.
    # PCA is used to reduce dimensionality and visualize class separability.
    #
    # Args:
    #     X_val (array-like): The validation data features.
    #     y_pred (array-like): The predicted labels for the validation data.
    def plot_pca(self, X_val, y_pred):
        # Scale features before applying PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_val)

        # Scale the PCA output to the [-1, 1] range for visualization
        X_pca_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_pca)
        
        df_pca = pd.DataFrame(data=X_pca_scaled, columns=['Principal Component 1', 'Principal Component 2'])
        df_pca['Predicted Label'] = y_pred
        
        n_classes = len(np.unique(y_pred))

        plt.figure(figsize=(11, 9))
        sns.scatterplot(
            x='Principal Component 1', y='Principal Component 2',
            hue='Predicted Label',
            palette=sns.color_palette("hsv", n_classes),
            data=df_pca,
            legend="full",
            alpha=0.8
        )
        
        variance_ratio = pca.explained_variance_ratio_
        plt.title('PCA of Predicted Gestures on Validation Set')
        plt.xlabel(f'Principal Component 1 (explains {variance_ratio[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 (explains {variance_ratio[1]:.2%} variance)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()


    # Generates and plots the t-SNE of the validation data colored by predicted labels.
    # t-SNE is used for visualizing high-dimensional data by giving each datapoint a location in a 2D map.
    #
    # Args:
    #     X_val (array-like): The validation data features.
    #     y_pred (array-like): The predicted labels for the validation data.
    def plot_tsne(self, X_val, y_pred):
        # Scale features before applying t-SNE
        X_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_val)
        
        # Perplexity should be less than the number of samples
        perplexity = min(30, len(X_scaled) - 1)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_scaled)
        
        df_tsne = pd.DataFrame(data=X_tsne, columns=['t-SNE Component 1', 't-SNE Component 2'])
        df_tsne['Predicted Label'] = y_pred
        
        n_classes = len(np.unique(y_pred))

        plt.figure(figsize=(11, 9))
        sns.scatterplot(
            x='t-SNE Component 1', y='t-SNE Component 2',
            hue='Predicted Label',
            palette=sns.color_palette("hsv", n_classes),
            data=df_tsne,
            legend="full",
            alpha=0.8
        )
        
        plt.title('t-SNE of Predicted Gestures on Validation Set')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.show()
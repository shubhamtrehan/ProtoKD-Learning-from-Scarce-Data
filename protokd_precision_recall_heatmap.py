import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def generate_precision_recall_heatmap(num_samples, csv_file_path, output_png_path):
    """
    Generates a PNG file containing class-wise metrics (precision, recall, F1-score) in a table,
    followed by average precision and recall, and a heatmap of the confusion matrix.

    Args:
        num_samples (int): Number of samples per class.
        csv_file_path (str): Path to the CSV file containing GroundTruth_Label and Predicted_label_name columns.
        output_png_path (str): Path to save the generated PNG file.

    Returns:
        None
    """
    # Load the CSV file
    data = pd.read_csv(csv_file_path)

    # Specify the column names
    ground_truth_column = "GroundTruth_Label"
    predicted_label_column = "Predicted_label_name"

    # Calculate classification metrics using sklearn's classification_report
    classes = sorted(data[ground_truth_column].unique())
    report = classification_report(
        data[ground_truth_column],
        data[predicted_label_column],
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()

    # Separate overall accuracy, macro avg, and weighted avg from per-class metrics
    overall_metrics = report_df.loc[["accuracy", "macro avg", "weighted avg"]]
    class_metrics = report_df.drop(index=["accuracy", "macro avg", "weighted avg"])

    # Extract average precision and recall
    average_precision = overall_metrics.loc["macro avg", "precision"]
    average_recall = overall_metrics.loc["macro avg", "recall"]

    # Create confusion matrix
    conf_matrix = confusion_matrix(data[ground_truth_column], data[predicted_label_column], labels=classes)

    # Plot the results
    fig, ax = plt.subplots(3, 1, figsize=(12, 20))

    # Title
    fig.suptitle(f"Precision and Recall for {num_samples} Samples Per Class", fontsize=16, x=0.1, ha='left')

    # Class-wise metrics table
    ax[0].axis('off')  # Remove the axes for a clean look
    table = ax[0].table(cellText=class_metrics.values.round(2),
                        colLabels=class_metrics.columns,
                        rowLabels=class_metrics.index,
                        cellLoc='center',
                        loc='upper left',
                        bbox=[0, 0, 1, 1])  # Adjust position and size of the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax[0].set_title("Class-wise Precision, Recall, and F1-Score", fontsize=14, loc='left', pad=10)

    # Average precision and recall
    ax[1].axis('off')  # Remove axes for text
    avg_text = f"Average Precision (Macro): {average_precision:.2f}\nAverage Recall (Macro): {average_recall:.2f}"
    ax[1].text(0, 0.5, avg_text, fontsize=12, va='center', ha='left')
    ax[1].set_title("Average Precision and Recall", fontsize=14, loc='left', pad=10)

    # Heatmap
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues", ax=ax[2])
    ax[2].set_title("Confusion Matrix Heatmap", fontsize=14)
    ax[2].set_xlabel("Predicted Labels")
    ax[2].set_ylabel("True Labels")

    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_png_path)

    plt.show()

    print(f"Precision, recall, and heatmap saved to: {output_png_path}")


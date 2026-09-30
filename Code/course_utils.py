"""Small shared helpers; model fitting stays visible in each lesson."""
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             mean_absolute_error, mean_squared_error, r2_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
def repository_path(value):
    """Relative configuration paths are anchored to the repo, not process cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


OUTPUT = repository_path(os.environ.get("COURSE_OUTPUT_DIR", "outputs"))


def finish_plot(name, directory=None):
    """Save every figure; set COURSE_SHOW_PLOTS=1 for an interactive window."""
    directory = OUTPUT if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Figure: {path}")
    if os.environ.get("COURSE_SHOW_PLOTS") == "1":
        plt.show()
    plt.close()


def regression_report(y_true, y_pred):
    result = {"MAE": mean_absolute_error(y_true, y_pred),
              "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
              "R2": r2_score(y_true, y_pred)}
    print(result)
    return result


def classification_summary(model, X_test, y_test, name, class_names=None):
    predicted = model.predict(X_test)
    print(classification_report(y_test, predicted, labels=model.classes_,
                                target_names=class_names, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, labels=model.classes_,
                                            display_labels=class_names)
    finish_plot(name + "_confusion")
    return predicted


def decision_plot(model, X, y, name, labels):
    """Evaluate a 150x150 grid in the SAME coordinates used by model.predict.

    Passing a Pipeline and raw inputs keeps the axes in original units.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("decision_plot requires exactly two input features")
    classes = np.asarray(model.classes_)
    # One shared discrete palette: the background and each class's points agree.
    colors = plt.get_cmap("tab10")(np.arange(len(classes)))
    cmap = ListedColormap(colors)
    levels = np.arange(len(classes) + 1) - 0.5
    norm = BoundaryNorm(levels, cmap.N)
    span = np.ptp(values, axis=0)
    pad = np.maximum(span * 0.05, 0.1)
    x1, x2 = np.meshgrid(
        np.linspace(values[:, 0].min() - pad[0], values[:, 0].max() + pad[0], 150),
        np.linspace(values[:, 1].min() - pad[1], values[:, 1].max() + pad[1], 150))
    grid = np.column_stack([x1.ravel(), x2.ravel()])
    if hasattr(X, "columns"):
        import pandas as pd
        grid = pd.DataFrame(grid, columns=X.columns)
    predictions = model.predict(grid)
    z = np.searchsorted(classes, predictions).reshape(x1.shape)
    fig, ax = plt.subplots()
    ax.contourf(x1, x2, z, levels=levels, alpha=0.25, cmap=cmap, norm=norm)
    for index, label in enumerate(classes):
        mask = np.asarray(y) == label
        ax.scatter(values[mask, 0], values[mask, 1], label=str(label), color=colors[index], edgecolors="black", linewidths=0.3, s=18)
    ax.set(xlabel=labels[0], ylabel=labels[1], title=name)
    ax.legend(title="Class")
    finish_plot(name)

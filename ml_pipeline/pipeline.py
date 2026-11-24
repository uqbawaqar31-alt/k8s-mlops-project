import json
from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from kfp import dsl
    from kfp import compiler
    HAS_KFP = True
except ImportError:
    HAS_KFP = False


def load_and_preprocess(output_dir: str) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    prep_file = out_path / "preprocessed_iris.joblib"
    joblib.dump(
        {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
        },
        prep_file,
    )
    return str(prep_file)


def train_model(prep_file: str, output_dir: str) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = joblib.load(prep_file)
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(data["X_train"], data["y_train"])

    model_file = out_path / "iris_model.joblib"
    joblib.dump(model, model_file)
    return str(model_file)


def evaluate_model(prep_file: str, model_file: str, output_dir: str) -> str:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = joblib.load(prep_file)
    model = joblib.load(model_file)

    preds = model.predict(data["X_test"])
    acc = float(accuracy_score(data["y_test"], preds))

    metrics_file = out_path / "metrics.json"
    metrics = {
        "dataset": "iris",
        "model_type": "LogisticRegression",
        "accuracy": round(acc, 4),
    }
    metrics_file.write_text(json.dumps(metrics, indent=2))
    return str(metrics_file)


def compile_kfp_yaml(yaml_out: str) -> None:
    if not HAS_KFP:
        print("kfp package not installed; skipping YAML compilation.")
        return

    @dsl.component(base_image="python:3.11-slim", packages_to_install=["scikit-learn", "joblib"])
    def comp_preprocess(test_size: float = 0.2) -> dict:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        iris = load_iris()
        X_tr, X_te, y_tr, y_te = train_test_split(
            iris.data, iris.target, test_size=test_size, random_state=42, stratify=iris.target
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr).tolist()
        X_te = scaler.transform(X_te).tolist()
        return {"X_train": X_tr, "X_test": X_te, "y_train": y_tr.tolist(), "y_test": y_te.tolist()}

    @dsl.component(base_image="python:3.11-slim", packages_to_install=["scikit-learn"])
    def comp_train(dataset: dict) -> dict:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=200, random_state=42)
        clf.fit(dataset["X_train"], dataset["y_train"])
        return {
            "coef": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "classes": clf.classes_.tolist(),
            "X_test": dataset["X_test"],
            "y_test": dataset["y_test"],
        }

    @dsl.component(base_image="python:3.11-slim", packages_to_install=["scikit-learn"])
    def comp_evaluate(trained: dict) -> float:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        clf = LogisticRegression()
        clf.classes_ = np.array(trained["classes"])
        clf.coef_ = np.array(trained["coef"])
        clf.intercept_ = np.array(trained["intercept"])
        preds = clf.predict(trained["X_test"])
        return float(accuracy_score(trained["y_test"], preds))

    @dsl.pipeline(name="iris-classification-pipeline", description="Iris ML pipeline")
    def iris_pipeline(test_size: float = 0.2):
        step1 = comp_preprocess(test_size=test_size)
        step2 = comp_train(dataset=step1.output)
        comp_evaluate(trained=step2.output)

    compiler.Compiler().compile(pipeline_func=iris_pipeline, package_path=yaml_out)
    print(f"Compiled Kubeflow pipeline YAML: {yaml_out}")


def main():
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    yaml_path = Path(__file__).resolve().parent / "iris_pipeline.yaml"

    compile_kfp_yaml(str(yaml_path))

    prep_path = load_and_preprocess(str(artifacts_dir))
    model_path = train_model(prep_path, str(artifacts_dir))
    metrics_path = evaluate_model(prep_path, model_path, str(artifacts_dir))

    print("Step A (Preprocess):", prep_path)
    print("Step B (Model):     ", model_path)
    print("Step C (Metrics):   ", Path(metrics_path).read_text())


if __name__ == "__main__":
    main()

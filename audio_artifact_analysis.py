from dataclasses import dataclass
from pathlib import Path
from scipy.stats import mannwhitneyu

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %% Configuration

PRACTICAL_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ExperimentConfig:
    base_dir: Path = Path("dataset/LA")
    output_dir: Path = PRACTICAL_DIR
    random_seed: int = 42
    samples_per_class: int = 300
    sample_rate: int = 16000
    cv_splits: int = 5
    test_size: float = 0.2
    pitch_min_hz: float = 50.0
    pitch_max_hz: float = 500.0
    
    @property
    def protocol_file(self) -> Path:
        return (
            self.base_dir
            / "ASVspoof2019_LA_cm_protocols"
            / "ASVspoof2019.LA.cm.train.trn.txt"
            )
    
    @property
    def audio_dir(self) -> Path:
        return self.base_dir / "ASVspoof2019_LA_train" / "flac"
    
CONFIG = ExperimentConfig()
DATA_DIR = CONFIG.output_dir / "data"
TABLES_DIR = CONFIG.output_dir / "tables"
FIGURES_DIR = CONFIG.output_dir / "figures"

FEATURE_COLUMNS =[
    "mfcc_mean",
    "mfcc_std",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "spectral_flatness_mean",
    "energy_0_2khz",
    "energy_2_4khz",
    "energy_4_8khz",
    "pitch_mean_hz",
    "pitch_std_hz",
    "pitch_range_hz",
    "voiced_fraction",
    ]

PLOT_FEATURE_COLUMNS = [
    "mfcc_std",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "spectral_flatness_mean",
    ]

ML_FEATURE_COLUMNS = (
    FEATURE_COLUMNS
    + [f"mfcc_{index}_mean" for index in range(1, 14)]
    + [f"mfcc_{index}_std" for index in range(1, 14)]
    + [f"mfcc_delta_{index}_mean" for index in range(1, 14)]
    + [f"mfcc_delta_{index}_std" for index in range(1, 14)]
    )
#%% Logging

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        )
    
#%% Protocol loading

def load_asvspoof_protocol(config: ExperimentConfig) -> pd.DataFrame:
    records = []
    
    with config.protocol_file.open("r", encoding="utf-8") as protocol:
        for line in protocol:
            speaker_id, file_id, _, attack_id, label = line.strip().split()
            
            records.append(
                {
                    "speaker_id": speaker_id,
                    "file_id": file_id,
                    "attack_id": attack_id,
                    "label": label,
                    "audio_path": config.audio_dir / f"{file_id}.flac",
                    }
                )
    protocol_df = pd.DataFrame.from_records(records)
    
    missing_files = protocol_df[
        ~protocol_df["audio_path"].apply(lambda path: path.exists())
        ]
    
    if not missing_files.empty:
        raise FileNotFoundError(
            f"{len(missing_files)} audio files referenced in the protocol were not found."
            )
    return protocol_df

#%% Sampling

def create_balanced_sample(
        protocol_df: pd.DataFrame,
        config: ExperimentConfig,
        ) -> pd.DataFrame:
    bonafide = protocol_df[protocol_df["label"] == "bonafide"]
    spoof = protocol_df[protocol_df["label"] == "spoof"]
    
    if len(bonafide) < config.samples_per_class:
        raise ValueError("Not enough bonafide samples available.")
        
    if len(spoof) < config.samples_per_class:
        raise ValueError("Not enough spoof samples available")
        
    bonafide_sample = bonafide.sample(
        n=config.samples_per_class,
        random_state=config.random_seed,
        )
    
    spoof_sample = spoof.sample(
        n=config.samples_per_class,
        random_state=config.random_seed,
        )
    
    sample_df = pd.concat([bonafide_sample, spoof_sample], ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=config.random_seed)
    
    return sample_df.reset_index(drop=True)

#%% Feature extraction

def calculate_frequency_band_energy(
        y: np.ndarray,
        sample_rate: int,
        low_frequency: float,
        high_frequency: float,
        ) -> float:
    stft = librosa.stft(y)
    power_spectrum = np.abs(stft) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    
    total_energy = np.sum(power_spectrum)
    
    if total_energy == 0:
        return 0.0
    
    frequency_mask = (frequencies >= low_frequency) & (frequencies < high_frequency)
    band_energy = np.sum(power_spectrum[frequency_mask, :])
    
    return float(band_energy / total_energy)

def extract_audio_features(
        audio_path: Path,
        config: ExperimentConfig,
        ) -> dict:
    y, sample_rate = librosa.load(
        audio_path,
        sr=config.sample_rate,
        mono=True
        )
    mfcc = librosa.feature.mfcc(
        y=y,
        sr = sample_rate,
        n_mfcc=13,
        )
    mfcc_delta = librosa.feature.delta(mfcc)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sample_rate,
        )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sample_rate,
        )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sample_rate,
        )
    spectral_flatness = librosa.feature.spectral_flatness(
        y=y,
        )
    pitch_features = extract_pitch_features(
        y=y,
        sample_rate=sample_rate,
        config=config,
        )
    
    features = {
        "duration_seconds": librosa.get_duration(y=y, sr=sample_rate),
        "mfcc_mean": float(np.mean(mfcc)),
        "mfcc_std": float(np.std(mfcc)),
        "spectral_centroid_mean": float(np.mean(spectral_centroid)),
        "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
        "spectral_flatness_mean": float(np.mean(spectral_flatness)),
        "energy_0_2khz": calculate_frequency_band_energy(
            y,
            sample_rate,
            0,
            2000,
            ),
        "energy_2_4khz": calculate_frequency_band_energy(
            y,
            sample_rate,
            2000,
            4000,
            ),
        "energy_4_8khz": calculate_frequency_band_energy(
            y,
            sample_rate,
            4000,
            8000,
            ),
        **pitch_features,
    }
    
    for mfcc_index in range(mfcc.shape[0]):
        coefficient_number = mfcc_index + 1
            
        features[f"mfcc_{coefficient_number}_mean"] = float(
            np.mean(mfcc[mfcc_index])
            )
        features[f"mfcc_{coefficient_number}_std"] = float(
            np.std(mfcc[mfcc_index])
            )
        features[f"mfcc_delta_{coefficient_number}_mean"] = float(
            np.mean(mfcc_delta[mfcc_index])
            )
        features[f"mfcc_delta_{coefficient_number}_std"] = float(
            np.std(mfcc_delta[mfcc_index])
            )
    return features



def extract_features_for_sample(
        sample_df: pd.DataFrame,
        config: ExperimentConfig,
        ) -> pd.DataFrame:
    feature_records = []
    
    for index, row in sample_df.iterrows():
        logging.info(
            "Extracting features %s/%s: %s",
            index + 1,
            len(sample_df),
            row["file_id"],
            )
        audio_features = extract_audio_features(row["audio_path"], config)
        
        feature_records.append(
            {
                "speaker_id": row["speaker_id"],
                "file_id": row["file_id"],
                "attack_id": row["attack_id"],
                "label": row["label"],
                **audio_features,
                }
            )
    return pd.DataFrame.from_records(feature_records)

def calculate_rank_biserial_correlation(
        u_statistic: float,
        n_group_1: int,
        n_group_2: int,
        ) -> float:
    return (2 * u_statistic) / (n_group_1 * n_group_2) - 1
    
def run_statistical_tests(
        features_df: pd.DataFrame,
        feature_columns: list[str],
        ) -> pd.DataFrame:
    bonafide_df = features_df[features_df["label"] == "bonafide"]
    spoof_df = features_df[features_df["label"] == "spoof"]
    
    test_results = []
    
    for feature in feature_columns:
        bonafide_values = bonafide_df[feature].dropna()
        spoof_values = spoof_df[feature].dropna()
        
        test_result = mannwhitneyu(
            bonafide_values,
            spoof_values,
            alternative="two-sided",
            )
        
        effect_size = calculate_rank_biserial_correlation(
            test_result.statistic,
            len(bonafide_values),
            len(spoof_values),
            )
        
        test_results.append(
            {
                "feature": feature,
                "bonafide_mean": bonafide_values.mean(),
                "spoof_mean": spoof_values.mean(),
                "bonafide_median": bonafide_values.median(),
                "spoof_median": spoof_values.median(),
                "u_statistic": test_result.statistic,
                "p_value": test_result.pvalue,
                "rank_biserial_correlation": effect_size,
                "absolute_effect_size": abs(effect_size),
                }
            )
    return pd.DataFrame.from_records(test_results)


#%% Pitch Feature Function

def extract_pitch_features(
        y: np.ndarray,
        sample_rate: int,
        config: ExperimentConfig,
        ) -> dict:
    f0, voiced_flag, voiced_probabilities = librosa.pyin(
        y,
        fmin=config.pitch_min_hz,
        fmax=config.pitch_max_hz,
        sr=sample_rate,
        )
    valid_pitch_values = f0[~np.isnan(f0)]
    
    if len(valid_pitch_values) == 0:
        return {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_range_hz": 0.0,
            "voiced_fraction": 0.0,
            }
    return {
        "pitch_mean_hz": float(np.mean(valid_pitch_values)),
        "pitch_std_hz": float(np.std(valid_pitch_values)),
        "pitch_range_hz": float(np.max(valid_pitch_values)-np.min(valid_pitch_values)),
        "voiced_fraction": float(np.mean(voiced_flag)),
        }
#%% Visualization

def create_feature_boxplots(
        features_df: pd.DataFrame,
        feature_columns: list[str],
        output_dir: Path,
        ) -> None:
    for feature in feature_columns:
        figure, axis = plt.subplots(figsize=(7, 5))
        
        grouped_data = [
            features_df[features_df["label"] == "bonafide"][feature],
            features_df[features_df["label"] == "spoof"][feature],
            ]
        
        axis.boxplot(
            grouped_data,
            labels=["bonafide", "spoof"],
            showmeans=True,
            )
        
        axis.set_title(f"Distribution of {feature}")
        axis.set_xlabel("Class")
        axis.set_ylabel(feature)
        axis.grid(axis="y", linestyle="--", alpha=0.5)
        
        figure.tight_layout()
        
        output_path = output_dir / f"boxplot_{feature}.png"
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        
        logging.info("Boxplot saved to: %s", output_path)
        
def create_mel_spectrogram_comparison(
        sample_df: pd.DataFrame,
        config: ExperimentConfig,
        ) -> None:
    bonafide_row = sample_df[sample_df["label"] == "bonafide"].iloc[0]
    spoof_row = sample_df[sample_df["label"] == "spoof"].iloc[0]
    
    comparison_items = [
        ("bonafide", bonafide_row),
        ("spoof", spoof_row)
        ]
    
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12,5),
        sharey=True,
        constrained_layout=True,
        )
    
    for axis, (label, row) in zip(axes, comparison_items):
        y, sample_rate = librosa.load(
            row["audio_path"],
            sr=config.sample_rate,
            mono=True,
            )
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_mels=128,
            fmax=8000,
            )
        mel_spectrogram_db = librosa.power_to_db(
            mel_spectrogram,
            ref=np.max,
            )
        
        image = librosa.display.specshow(
            mel_spectrogram_db,
            sr=sample_rate,
            x_axis="time",
            y_axis="mel",
            fmax=8000,
            ax=axis,
            )
        axis.set_title(f"{label}: {row['file_id']}")
        axis.set_xlabel("Time")
        axis.set_ylabel("Mel frequency")
        
    figure.colorbar(
        image,
        ax=axes,
        format="%+2.0f dB",
        )
        
    figure.suptitle(
        "Mel-Spectrogram Comparison: Bonafide vs Spoof",
        fontsize=14,
        )
        
    output_path = FIGURES_DIR / "mel_spectrogram_comparison.png"
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
        
    logging.info("Mel-spectrogram comparison saved to: %s", output_path)
        

def create_pitch_contour_comparison(
        sample_df: pd.DataFrame,
        config: ExperimentConfig,
        ) -> None:
    bonafide_row = sample_df[sample_df["label"] == "bonafide"].iloc[0]
    spoof_row = sample_df[sample_df["label"] == "spoof"].iloc[0]
    
    comparison_items = [
        ("bonafide", bonafide_row),
        ("spoof", spoof_row),
        ]
    
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12,4),
        sharey=True,
        constrained_layout=True,
        )
    
    for axis, (label, row) in zip(axes, comparison_items):
        y, sample_rate = librosa.load(
            row["audio_path"],
            sr=config.sample_rate,
            mono=True,
            )
        
        f0, voiced_flag, voiced_probabilities = librosa.pyin(
            y,
            fmin=config.pitch_min_hz,
            fmax=config.pitch_max_hz,
            sr=sample_rate,
            )
        
        times = librosa.times_like(f0, sr=sample_rate)
        
        axis.plot(times, f0, linewidth=1.5)
        axis.set_title(f"{label}: {row['file_id']}")
        axis.set_xlabel("Time")
        axis.set_ylabel("F0 / Pitch in Hz")
        axis.set_ylim(config.pitch_min_hz, config.pitch_max_hz)
        axis.grid(True, linestyle="--", alpha=0.5)
        
        figure.suptitle(
            "Pitch Contour Comparison: Bonafide vs Spoof",
            fontsize=14,
            )
        
        output_path = FIGURES_DIR / "pitch_contour_comparison.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        
        logging.info("Pitch contour comparison saved to: %s", output_path)
        
#%% Classical Machine Learning Baseline

def prepare_ml_dataset(
        features_df: pd.DataFrame,
        feature_columns: list[str],
        ) -> tuple[pd.DataFrame, pd.Series]:
    x = features_df[feature_columns].copy()
    y = (features_df["label"] == "spoof").astype(int)
    return x, y

def build_ml_models(config: ExperimentConfig) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=config.random_seed,
                    ),
                ),
            ]
        ),
    
        "Support Vector Machine": Pipeline(
        steps= [
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=config.random_seed,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
   }     

def run_cross_validation(
        features_df: pd.DataFrame,
        feature_columns: list[str],
        config: ExperimentConfig,
        ) -> pd.DataFrame:
    x, y = prepare_ml_dataset(features_df, feature_columns)
    models = build_ml_models(config)
    
    cross_validator = StratifiedKFold(
        n_splits=config.cv_splits,
        shuffle=True,
        random_state=config.random_seed,
        )
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        }
    
    result_records = []
    
    for model_name, model in models.items():
        logging.info("Running %s with %s-fold cross-validation", model_name, config.cv_splits)
        
        scores = cross_validate(
            estimator=model,
            X=x,
            y=y,
            cv=cross_validator,
            scoring=scoring,
            n_jobs=1,
        )
        
        result_record = {"model": model_name}
        
        for metric_name in scoring:
            metric_values = scores[f"test_{metric_name}"]
            result_record[f"{metric_name}_mean"] = metric_values.mean()
            result_record[f"{metric_name}_std"] = metric_values.std()
            
        result_records.append(result_record)
    return pd.DataFrame.from_records(result_records)
            

def train_best_model_and_create_confusion_matrix(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: list[str],
        model_name: str,
        config: ExperimentConfig,
        ) -> None:
    x_train, y_train = prepare_ml_dataset(train_df, feature_columns)
    x_test, y_test = prepare_ml_dataset(test_df, feature_columns)
    selected_model = build_ml_models(config)[model_name]

    selected_model.fit(x_train, y_train)
    y_prediction = selected_model.predict(x_test)

    report = classification_report(
        y_test,
        y_prediction,
        target_names=["bonafide", "spoof"],
        output_dict=True,
        )

    report_df = pd.DataFrame(report).transpose()
    report_path = TABLES_DIR / "ml_classification_report.csv"
    report_df.to_csv(report_path)
    
    matrix = confusion_matrix(y_test, y_prediction)
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_bonafide", "actual_spoof"],
        columns=["predicted_bonafide", "predicted_spoof"],
        )
    
    matrix_path = TABLES_DIR / "ml_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)
    
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["bonafide", "spoof"],
        )
    
    figure, axis = plt.subplots(figsize=(6,5))
    display.plot(ax=axis, cmap="Blues", values_format="d")
    axis.set_title(f"Confusion Matrix: {model_name}")
    
    figure.tight_layout()
    
    output_path = FIGURES_DIR / "ml_confusion_matrix.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    
    logging.info("Classification report saved to: %s", report_path)
    logging.info("Confusion matrix saved to: %s", matrix_path)
    logging.info("Confusion matrix plot saved to: %s", output_path)
    
def train_random_forest_and_create_feature_importance(
        features_df: pd.DataFrame,
        feature_columns: list[str],
        config: ExperimentConfig,
        top_n: int = 15,
        ) -> None:
    x, y = prepare_ml_dataset(features_df, feature_columns)
    
    random_forest = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=config.random_seed,
        )
    random_forest.fit(x, y)
    
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": random_forest.feature_importances_,
            }
        )
    
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False,
        ).reset_index(drop=True)
    
    importance_path = TABLES_DIR / "random_forest_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    top_features_df = importance_df.head(top_n).sort_values(
        by="importance",
        ascending=True,
        )
    
    figure, axis = plt.subplots(figsize=(9, 6))
    
    axis.barh(
        top_features_df["feature"],
        top_features_df["importance"],
        )

    axis.set_title(f"Top {top_n} Random Forest Feature Importances")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.grid(axis="x", linestyle="--", alpha=0.5)

    figure.tight_layout()

    output_path = FIGURES_DIR / "random_forest_feature_importance.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    logging.info("Random Forest feature importance saved to: %s", importance_path)
    logging.info("Random Forest feature importance plot saved to: %s", output_path)
    logging.info("Top Random Forest features:\n%s", importance_df.head(top_n))
    
#%% Main checkpoint

def main() -> None:
    setup_logging()
    
    CONFIG.output_dir.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    
    logging.info("Loading ASVspoof protocol")
    protocol_df = load_asvspoof_protocol(CONFIG)
    
    logging.info("Protocol loaded")
    logging.info("Label distribution:\n%s", protocol_df["label"].value_counts())
    
    logging.info("Creating balanced sample")
    sample_df = create_balanced_sample(protocol_df, CONFIG)
    
    sample_path = DATA_DIR / "selected_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    
    logging.info("Selected sample saved to: %s", sample_path)
    logging.info("Sample distribution:\n%s", sample_df["label"].value_counts())
    
    logging.info("Extracting audio features")
    features_df = extract_features_for_sample(sample_df, CONFIG)
    
    features_path = DATA_DIR / "audio_features.csv"
    features_df.to_csv(features_path, index=False)
    
    logging.info("Audio features saved to: %s", features_path)
    
    summary_path = TABLES_DIR / "summary_by_label.csv"
    summary_df = features_df.groupby("label").mean(numeric_only=True)
    summary_df.to_csv(summary_path)
    
    logging.info("Summary by label saved to: %s", summary_path)
    logging.info("Summary by label: \n%s", summary_df)
    
    logging.info("Running statistical tests")
    statistical_tests_df = run_statistical_tests(features_df, FEATURE_COLUMNS)
    
    statistical_tests_path = TABLES_DIR / "statistical_tests.csv"
    statistical_tests_df.to_csv(statistical_tests_path, index=False)
    
    logging.info("Statistical tests saved to: %s", statistical_tests_path)
    logging.info("Statistical test results:\n%s", statistical_tests_df)
    
    logging.info("Creating feature boxplots")
    create_feature_boxplots(
        features_df=features_df,
        feature_columns=PLOT_FEATURE_COLUMNS,
        output_dir=FIGURES_DIR,
        )
    
    logging.info("Creating Mel-spectrogram comparison")
    create_mel_spectrogram_comparison(
        sample_df=sample_df,
        config=CONFIG,
        )

    logging.info("Creating pitch contour comparison")
    create_pitch_contour_comparison(
        sample_df=sample_df,
        config=CONFIG,
        )
    
    # Split before cross-validation and model selection to preserve an untouched test set.
    train_df, test_df = train_test_split(
        features_df,
        test_size=CONFIG.test_size,
        stratify=features_df["label"],
        random_state=CONFIG.random_seed,
    )
    logging.info("Training samples: %s; held-out test samples: %s", len(train_df), len(test_df))

    logging.info("Running classical ML baseline on training data only")
    ml_results_df = run_cross_validation(
        features_df=train_df,
        feature_columns=ML_FEATURE_COLUMNS,
        config=CONFIG,
        )
    
    ml_results_path = TABLES_DIR / "ml_cross_validation_results.csv"
    ml_results_df.to_csv(ml_results_path, index=False)
    
    logging.info("ML cross-validation results saved to: %s", ml_results_path)
    logging.info("ML cross-validation results:\n%s", ml_results_df)
    
    best_model_name = ml_results_df.sort_values(
        by="f1_mean",
        ascending=False,
        ).iloc[0]["model"]
    
    logging.info("Best model according to F1-score: %s", best_model_name)
    
    train_best_model_and_create_confusion_matrix(
        train_df=train_df,
        test_df=test_df,
        feature_columns=ML_FEATURE_COLUMNS,
        model_name=best_model_name,
        config=CONFIG,
        )
    
    logging.info("Creating Random Forest feature importance analysis")
    train_random_forest_and_create_feature_importance(
        features_df=train_df,
        feature_columns=ML_FEATURE_COLUMNS,
        config=CONFIG,
        top_n=15,
        )
    
if __name__ == "__main__":
    main()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

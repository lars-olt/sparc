"""Input/output utilities for SPARC."""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Union, Optional
from datetime import datetime

from ..core.constants import COLOR_NAMES


class SparcExporter:
    """
    Comprehensive exporter for SPARC results including CSV, plots, and images.
    """

    def __init__(self, sparc_instance):
        """
        Initialize exporter with SPARC instance.

        Args:
            sparc_instance: Completed SPARC pipeline instance
        """
        self.sparc = sparc_instance
        self.timestamp = datetime.now()

        if not sparc_instance.is_complete:
            raise ValueError("SPARC pipeline must be completed before export")

    def save_roi_csv(
        self, output_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save ROI results to CSV file in marslab format.

        Args:
            output_path: Output CSV file path
            metadata: Optional metadata dictionary with scene information
        """
        output_path = Path(output_path).with_suffix(".csv")

        # Default metadata
        default_metadata = {
            "SOL": 613,  # Default, should be provided
            "SEQ_ID": (
                self.sparc.load_result["id"] if self.sparc.load_result else "unknown"
            ),
            "ANALYSIS_NAME": "SPARC_ROI_ANALYSIS",
            "SITE": 0,
            "DRIVE": 0,
            "RSM": 0,
            "LTST": "00:00:00",
            "INCIDENCE_ANGLE": 45.0,
            "EMISSION_ANGLE": 0.0,
            "PHASE_ANGLE": 45.0,
            "SOLAR_ELEVATION": 45.0,
            "SOLAR_AZIMUTH": 180.0,
            "LAT": 0.0,
            "LON": 0.0,
            "ODOMETRY": 0.0,
            "ROVER_ELEVATION": 0.0,
            "SCLK": 0.0,
            "FEATURE": "rock",
            "FEATURE_SUBTYPE": "unknown",
            "DESCRIPTION": "ROI identified by SPARC",
            "ROI_SOURCE": "SPARC_AUTO",
            "FORMATION": "unknown",
            "GRAIN_SIZE": "unknown",
            "MEMBER": "unknown",
            "DISTANCE": "unknown",
            "ZOOM": 1,
            "COMPRESSION": "none",
            "L_S": 0.0,
            "COMPRESSION_QUALITY": 100,
            "CREATOR": "SPARC",
            "RC_SEL_FILE": "",
            "RC_CALTARGET_FILE": "",
            "RC_SOL": 613,
            "RC_SEQ_ID": "",
            "RC_LTST": "00:00:00",
            "RC_SOLAR_AZIMUTH": 180.0,
            "RC_SCALING_FACTOR": 1.0,
            "RC_UNCERTAINTY": 0.05,
            "RC_AZIMUTH_ANGLE": 0.0,
            "RC_EMISSION_ANGLE": 0.0,
            "RC_INCIDENCE_ANGLE": 45.0,
            "FILE_TIMESTAMP": self.timestamp.isoformat(),
        }

        # Update with provided metadata
        if metadata:
            default_metadata.update(metadata)

        # Get final ROI data
        final_rois = self.sparc.final_rois

        # Get corresponding spectra
        if len(self.sparc.roi_spectra[self.sparc.outlier_mask]) > 3:
            outlier_spectra = self.sparc.roi_spectra[self.sparc.outlier_mask]
            outlier_stds = self.sparc.roi_stds[self.sparc.outlier_mask]
            final_spectra = outlier_spectra[self.sparc.roi_indices]
            final_stds = outlier_stds[self.sparc.roi_indices]
        else:
            final_spectra = self.sparc.roi_spectra[self.sparc.roi_indices]
            final_stds = self.sparc.roi_stds[self.sparc.roi_indices]

        # Create DataFrame
        rows = []

        # Band names mapping from WLS to CSV column names
        band_mapping = {
            0: ("L0B", "R0B"),  # Blue Bayer
            1: ("L0G", "R0G"),  # Green Bayer
            2: ("L0R", "R0R"),  # Red Bayer
            3: ("L1", "R1"),  # 800nm
            4: ("L2", "R2"),  # 754nm
            5: ("L3", "R3"),  # 677nm
            6: ("L4", "R4"),  # 605nm
            7: ("L5", "R5"),  # 528nm
            8: ("L6", "R6"),  # 442nm
            9: ("R6", None),  # 866nm (right only)
            10: ("R5", None),  # 910nm (right only)
            11: ("R4", None),  # 939nm (right only)
            12: ("R3", None),  # 978nm (right only)
            13: ("R2", None),  # 1022nm (right only)
        }

        for i, (roi, spectrum, std) in enumerate(
            zip(final_rois, final_spectra, final_stds)
        ):
            x, y, width, height = roi

            # Basic ROI info
            row_data = default_metadata.copy()
            row_data.update(
                {
                    "NAME": f"ROI_{i+1:03d}",
                    "COLOR": f"{COLOR_NAMES[i % len(COLOR_NAMES)]}",  # Cycle through colors
                    "ROW": float(y + height / 2),  # Center row
                    "COLUMN": float(x + width / 2),  # Center column
                    "DET_RAD": float(
                        np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
                    ),  # Radius
                    "DET_THETA": 0.0,  # Angle
                    "LEFT_DET_RAD": float(
                        np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
                    ),
                    "LEFT_DET_THETA": 0.0,
                    "LEFT_COUNT": float(width * height),
                    "RIGHT_DET_RAD": float(
                        np.sqrt((width / 2) ** 2 + (height / 2) ** 2)
                    ),
                    "RIGHT_DET_THETA": 0.0,
                    "RIGHT_COUNT": float(width * height),
                    "FLOAT": f"ROI_{i+1:03d}",
                }
            )

            # Add spectral data
            for band_idx, (left_col, right_col) in band_mapping.items():
                if band_idx < len(spectrum):
                    value = float(spectrum[band_idx])
                    std_value = float(std[band_idx])
                    count_value = float(width * height)  # Use ROI area as count

                    # Add left channel data
                    if left_col:
                        row_data[left_col] = value
                        row_data[f"{left_col}_STD"] = std_value
                        row_data[f"{left_col}_COUNT"] = count_value

                    # Add right channel data (if exists)
                    if right_col:
                        row_data[right_col] = value
                        row_data[f"{right_col}_STD"] = std_value
                        row_data[f"{right_col}_COUNT"] = count_value

            rows.append(row_data)

        # Create DataFrame and save
        df = pd.DataFrame(rows)

        # Ensure all expected columns are present
        expected_columns = [
            "NAME",
            "COLOR",
            "ANALYSIS_NAME",
            "SOL",
            "SEQ_ID",
            "FEATURE",
            "FEATURE_SUBTYPE",
            "DESCRIPTION",
            "SITE",
            "DRIVE",
            "RSM",
            "LTST",
            "INCIDENCE_ANGLE",
            "EMISSION_ANGLE",
            "PHASE_ANGLE",
            "SOLAR_ELEVATION",
            "SOLAR_AZIMUTH",
            "LAT",
            "LON",
            "ODOMETRY",
            "ROVER_ELEVATION",
            "SCLK",
            "L6",
            "L0B",
            "R0B",
            "L5",
            "L0G",
            "R0G",
            "L4",
            "L0R",
            "R0R",
            "L3",
            "L2",
            "L1",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "R6",
            "L6_STD",
            "L0B_STD",
            "R0B_STD",
            "L5_STD",
            "L0G_STD",
            "R0G_STD",
            "L4_STD",
            "L0R_STD",
            "R0R_STD",
            "L3_STD",
            "L2_STD",
            "L1_STD",
            "R1_STD",
            "R2_STD",
            "R3_STD",
            "R4_STD",
            "R5_STD",
            "R6_STD",
            "L6_COUNT",
            "L0B_COUNT",
            "R0B_COUNT",
            "L5_COUNT",
            "L0G_COUNT",
            "R0G_COUNT",
            "L4_COUNT",
            "L0R_COUNT",
            "R0R_COUNT",
            "L3_COUNT",
            "L2_COUNT",
            "L1_COUNT",
            "R1_COUNT",
            "R2_COUNT",
            "R3_COUNT",
            "R4_COUNT",
            "R5_COUNT",
            "R6_COUNT",
            "LEFT_DET_RAD",
            "LEFT_DET_THETA",
            "LEFT_COUNT",
            "RIGHT_DET_RAD",
            "RIGHT_DET_THETA",
            "RIGHT_COUNT",
            "ROW",
            "COLUMN",
            "DET_RAD",
            "DET_THETA",
            "ROI_SOURCE",
            "FLOAT",
            "FORMATION",
            "GRAIN_SIZE",
            "MEMBER",
            "DISTANCE",
            "ZOOM",
            "COMPRESSION",
            "L_S",
            "COMPRESSION_QUALITY",
            "CREATOR",
            "RC_SEL_FILE",
            "RC_CALTARGET_FILE",
            "RC_SOL",
            "RC_SEQ_ID",
            "RC_LTST",
            "RC_SOLAR_AZIMUTH",
            "RC_SCALING_FACTOR",
            "RC_UNCERTAINTY",
            "RC_AZIMUTH_ANGLE",
            "RC_EMISSION_ANGLE",
            "RC_INCIDENCE_ANGLE",
            "FILE_TIMESTAMP",
        ]

        # Reorder columns to match expected format
        df = df.reindex(columns=expected_columns, fill_value=0.0)

        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"ROI CSV saved to: {output_path}")

    def save_spectra_plot(
        self,
        output_path: Union[str, Path],
        figsize: tuple = (10, 8),
        dpi: int = 300,
        format: str = "png",
    ) -> None:
        """
        Save the spectra plot to file.

        Args:
            output_path: Output image file path
            figsize: Figure size (width, height)
            dpi: Image resolution
            format: Image format ('png', 'pdf', 'svg', etc.)
        """
        output_path = Path(output_path).with_suffix(f".{format}")

        # Get final spectra
        if len(self.sparc.roi_spectra[self.sparc.outlier_mask]) > 3:
            outlier_spectra = self.sparc.roi_spectra[self.sparc.outlier_mask]
            outlier_stds = self.sparc.roi_stds[self.sparc.outlier_mask]
            final_spectra = outlier_spectra[self.sparc.roi_indices]
            final_stds = outlier_stds[self.sparc.roi_indices]
        else:
            final_spectra = self.sparc.roi_spectra[self.sparc.roi_indices]
            final_stds = self.sparc.roi_stds[self.sparc.roi_indices]

        # Create plot
        from ..visualization.plotting import plot_spectra

        fig = plot_spectra(final_spectra, final_stds, show=False)

        # Set figure size and save
        fig.set_size_inches(figsize)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format=format)
        plt.close(fig)

        print(f"Spectra plot saved to: {output_path}")

    def save_roi_context_image(
        self,
        output_path: Union[str, Path],
        figsize: tuple = (12, 9),
        dpi: int = 300,
        format: str = "png",
    ) -> None:
        """
        Save the ROI context image (RGB image with ROI overlays) to file.

        Args:
            output_path: Output image file path
            figsize: Figure size (width, height)
            dpi: Image resolution
            format: Image format ('png', 'pdf', 'svg', etc.)
        """
        output_path = Path(output_path).with_suffix(f".{format}")

        # Create plot
        from ..visualization.plotting import plot_roi_image

        fig = plot_roi_image(
            self.sparc.load_result["rgb_img"], self.sparc.final_rois, show=False
        )

        # Set figure size and save
        fig.set_size_inches(figsize)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format=format)
        plt.close(fig)

        print(f"ROI context image saved to: {output_path}")

    def save_pipeline_summary_image(
        self,
        output_path: Union[str, Path],
        figsize: tuple = (15, 12),
        dpi: int = 300,
        format: str = "png",
    ) -> None:
        """
        Save the complete pipeline summary plot to file.

        Args:
            output_path: Output image file path
            figsize: Figure size (width, height)
            dpi: Image resolution
            format: Image format ('png', 'pdf', 'svg', etc.)
        """
        output_path = Path(output_path).with_suffix(f".{format}")

        # Create plot
        from ..visualization.plotting import plot_pipeline_summary

        fig = plot_pipeline_summary(self.sparc, show=False)

        # Set figure size and save
        fig.set_size_inches(figsize)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", format=format)
        plt.close(fig)

        print(f"Pipeline summary saved to: {output_path}")

    def export_complete_results(
        self,
        output_dir: Union[str, Path],
        base_name: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """
        Export all results (CSV, plots, images) to a directory.

        Args:
            output_dir: Output directory path
            base_name: Base name for output files (default: scene_id)
            metadata: Optional metadata for CSV export

        Returns:
            Dictionary mapping export type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if base_name is None:
            base_name = (
                self.sparc.load_result["id"]
                if self.sparc.load_result
                else "sparc_results"
            )

        exported_files = {}

        # Save CSV
        csv_path = output_dir / f"{base_name}_rois.csv"
        self.save_roi_csv(csv_path, metadata)
        exported_files["csv"] = csv_path

        # Save spectra plot
        spectra_path = output_dir / f"{base_name}_spectra.png"
        self.save_spectra_plot(spectra_path)
        exported_files["spectra_plot"] = spectra_path

        # Save ROI context image
        context_path = output_dir / f"{base_name}_roi_context.png"
        self.save_roi_context_image(context_path)
        exported_files["roi_context"] = context_path

        # Save pipeline summary
        summary_path = output_dir / f"{base_name}_pipeline_summary.png"
        self.save_pipeline_summary_image(summary_path)
        exported_files["pipeline_summary"] = summary_path

        # Save results pickle for later loading
        pickle_path = output_dir / f"{base_name}_results.pkl"
        save_sparc_results(self.sparc, pickle_path, "pickle")
        exported_files["results_pickle"] = pickle_path

        print(f"\nComplete results exported to: {output_dir}")
        print("Files created:")
        for export_type, file_path in exported_files.items():
            print(f"  {export_type}: {file_path.name}")

        return exported_files


def save_sparc_results(
    sparc_instance, output_path: Union[str, Path], save_format: str = "pickle"
) -> None:
    """
    Save SPARC results to file.

    Args:
        sparc_instance: Completed SPARC instance
        output_path: Output file path
        save_format: Format to save ('pickle', 'json', 'npz')
    """
    output_path = Path(output_path)

    # Prepare data for saving
    results = {
        "scene_id": (
            sparc_instance.load_result["id"] if sparc_instance.load_result else None
        ),
        "final_rois": (
            sparc_instance.final_rois.tolist()
            if sparc_instance.final_rois is not None
            else None
        ),
        "roi_spectra": (
            sparc_instance.roi_spectra.tolist()
            if sparc_instance.roi_spectra is not None
            else None
        ),
        "roi_stds": (
            sparc_instance.roi_stds.tolist()
            if sparc_instance.roi_stds is not None
            else None
        ),
        "clustering_result": sparc_instance.clustering_result,
        "config": sparc_instance.config,
        "summary": sparc_instance.summary,
    }

    if save_format == "pickle":
        with open(output_path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(results, f)

    elif save_format == "json":
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj

        json_results = json.loads(json.dumps(results, default=convert_numpy))
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(json_results, f, indent=2)

    elif save_format == "npz":
        np.savez_compressed(
            output_path.with_suffix(".npz"),
            final_rois=sparc_instance.final_rois or np.array([]),
            roi_spectra=sparc_instance.roi_spectra or np.array([]),
            roi_stds=sparc_instance.roi_stds or np.array([]),
            **{
                k: v
                for k, v in results.items()
                if k not in ["final_rois", "roi_spectra", "roi_stds"]
            },
        )

    else:
        raise ValueError(f"Unsupported save format: {save_format}")


def load_sparc_results(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load SPARC results from file.

    Args:
        file_path: Path to results file

    Returns:
        Dictionary of loaded results
    """
    file_path = Path(file_path)

    if file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            return pickle.load(f)

    elif file_path.suffix == ".json":
        with open(file_path, "r") as f:
            return json.load(f)

    elif file_path.suffix == ".npz":
        data = np.load(file_path, allow_pickle=True)
        return {key: data[key] for key in data.keys()}

    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def export_rois_to_csv(
    rois: np.ndarray, spectra: np.ndarray, output_path: Union[str, Path]
) -> None:
    """
    Export ROIs and spectra to simple CSV format.

    Args:
        rois: ROI coordinates
        spectra: ROI spectra
        output_path: Output CSV file path
    """
    # Create DataFrame with ROI coordinates
    roi_df = pd.DataFrame(rois, columns=["x", "y", "width", "height"])
    roi_df["roi_id"] = range(len(rois))

    # Add spectral data
    spectral_columns = [f"band_{i}" for i in range(spectra.shape[1])]
    spectra_df = pd.DataFrame(spectra, columns=spectral_columns)

    # Combine
    combined_df = pd.concat([roi_df, spectra_df], axis=1)
    combined_df.to_csv(output_path, index=False)

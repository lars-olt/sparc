import numpy as np
from pathlib import Path
from sparc import Sparc


def main():
    sam_model_path = "[replace with path to model]/sam_vit_h_4b8939.pth"
    products_path = "[replace with path to products]"

    root_dir = Path(products_path)  # directory containing sol data directories
    sols = np.array([f.name for f in root_dir.iterdir() if f.is_dir()])

    for sol in sols:
        try:
            for obs_ix in range(30):
                sparc = Sparc(sam_model_path)
                sparc.run_pipeline(iof_path=root_dir / f"{sol}/iof", obs_ix=obs_ix)
                sparc.export_results(
                    f"results/{sol}/{obs_ix}", metadata={"SOL": int(sol)}
                )
                del sparc
        except:
            del sparc
            continue  # found all obs_ix's for scene


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse

from hireflow.ingestion import ingest_resumes


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse, embed, and index resumes.")
    parser.add_argument(
        "--resume-dir",
        default="resume_dir",
        help="Folder containing PDF, DOCX, TXT, or MD resumes.",
    )
    args = parser.parse_args()

    count = ingest_resumes(Path(args.resume_dir))
    print(f"Indexed {count} resumes.")


if __name__ == "__main__":
    main()


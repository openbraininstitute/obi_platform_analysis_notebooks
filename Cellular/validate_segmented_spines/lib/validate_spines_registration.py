"""
registration.py

Core logic for experiment data creation, packaging, and upload
to Google Drive through a Google Apps Script endpoint.
"""

from pathlib import Path
from datetime import datetime

import base64
import mimetypes
import shutil
import zipfile

import numpy as np
import pandas as pd
import requests
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

UPLOAD_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbyG2aSfIs7coMb_Ry7G9Jkppj7p0rtn2DBLL47_kIkYA3FN0g66OKmX87DpmRxxXyHf/exec"
)

DEFAULT_DRIVE_FOLDER = "archives"
UPLOAD_TIMEOUT_SECONDS = 60 * 60

# =============================================================================
# Timestamp / identifier
# =============================================================================

def get_timestamp(time_stamp: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string into a datetime object.

    Example:
        2026-08-24T15:14:35Z
    """

    return datetime.fromisoformat(
        time_stamp.replace("Z", "+00:00")
    )


def build_identifier(
    neuron_pt_root_id: str,
    user_name: str,
    dt: datetime,
) -> str:
    """
    Build a filesystem-friendly unique experiment identifier.

    Example:
        864691135123_marwan_20260824_151435
    """

    timestamp = dt.strftime("%Y%m%d_%H%M%S")

    safe_user_name = (
        user_name
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return (
        f"{neuron_pt_root_id}_"
        f"{safe_user_name}_"
        f"{timestamp}"
    )


# =============================================================================
# Data creation
# =============================================================================

def create_images(
    output_dir: Path,
    n_images: int = 3,
    image_size: tuple[int, int] = (512, 512),
) -> None:
    """
    Generate random RGB images and save them as PNG files.
    """

    width, height = image_size

    print(
        f"[Generation] Generating {n_images} "
        f"images ({width}x{height})..."
    )

    for i in range(n_images):

        data = np.random.randint(
            0,
            256,
            size=(height, width, 3),
            dtype=np.uint8,
        )

        path = (
            output_dir /
            f"image_{i + 1}.png"
        )

        Image.fromarray(
            data
        ).save(
            path
        )

        print(
            f"[Generation]   ✓ {path.name}"
        )


def create_csv(
    output_dir: Path,
    n_rows: int = 50,
) -> None:
    """
    Generate a CSV file containing random experiment metrics.
    """

    print(
        f"[Generation] Generating CSV "
        f"with {n_rows} rows..."
    )

    df = pd.DataFrame(
        {
            "neuron_id": np.arange(n_rows),
            "signal": np.random.rand(n_rows),
            "noise": np.random.rand(n_rows),
            "snr": np.random.rand(n_rows) * 10,
        }
    )

    path = (
        output_dir /
        "metrics.csv"
    )

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"[Generation]   ✓ {path.name} "
        f"({n_rows} rows, {len(df.columns)} columns)"
    )


def create_experiment_data(
    identifier: str,
    summary_text: str,
    n_images: int = 3,
    image_size: tuple[int, int] = (512, 512),
    n_csv_rows: int = 50,
) -> Path:
    """
    Create an experiment directory containing:

        summary.txt
        image_1.png
        image_2.png
        ...
        metrics.csv

    Returns:
        Path to the generated experiment directory.
    """

    print(
        f"[Generation] Starting data generation "
        f"for '{identifier}'..."
    )

    output_dir = Path(
        identifier
    )

    if output_dir.exists():

        print(
            f"[Generation] Removing existing "
            f"directory '{output_dir}'..."
        )

        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary_path = (
        output_dir /
        "summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    print(
        "[Generation]   ✓ summary.txt"
    )


    # -------------------------------------------------------------------------
    # Images
    # -------------------------------------------------------------------------

    create_images(
        output_dir,
        n_images=n_images,
        image_size=image_size,
    )


    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    create_csv(
        output_dir,
        n_rows=n_csv_rows,
    )


    print(
        f"[Generation] Data generation complete "
        f"→ {output_dir}/"
    )

    return output_dir


# =============================================================================
# Packaging
# =============================================================================

def create_zip(
    output_dir: Path,
) -> Path:
    """
    Zip the contents of an experiment directory.

    Example:

        experiment_001/

    becomes:

        experiment_001.zip

    Returns:
        Path to the generated ZIP file.
    """

    output_dir = Path(
        output_dir
    )

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist: "
            f"{output_dir}"
        )

    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory: {output_dir}"
        )

    zip_path = output_dir.parent / f'{output_dir.name}.zip'

    print(
        f"[Zipping] Compressing "
        f"'{output_dir}' → '{zip_path}'..."
    )

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zip_file:

        for file_path in output_dir.rglob("*"):

            if not file_path.is_file():
                continue

            archive_name = (
                file_path.relative_to(
                    output_dir
                )
            )

            zip_file.write(
                file_path,
                arcname=archive_name,
            )

    zip_size_mb = (
        zip_path.stat().st_size /
        (1024 * 1024)
    )

    print(
        f"[Zipping] ✓ Done → {zip_path} "
        f"({zip_size_mb:.2f} MB)"
    )

    return zip_path


# =============================================================================
# Google Drive upload
# =============================================================================

def upload_file(
    file_path: Path,
    folder_name: str,
    message: str | None = None,
) -> dict:
    """
    Upload a file to Google Drive through the Apps Script endpoint.

    Args:
        file_path:
            Local file to upload.

        folder_name:
            Destination folder name on Google Drive.

        message:
            Optional custom text that will be included
            in the notification email sent by Apps Script.

    Returns:
        JSON response returned by Apps Script.
    """

    file_path = Path(
        file_path
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"File does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Expected a file: {file_path}"
        )


    # -------------------------------------------------------------------------
    # Determine MIME type
    # -------------------------------------------------------------------------

    mime_type, _ = mimetypes.guess_type(
        file_path
    )

    if mime_type is None:
        mime_type = (
            "application/octet-stream"
        )


    # -------------------------------------------------------------------------
    # Read and Base64 encode file
    # -------------------------------------------------------------------------

    file_size_mb = (
        file_path.stat().st_size /
        (1024 * 1024)
    )

    print(
        f"[Upload] Preparing '{file_path.name}' "
        f"({file_size_mb:.2f} MB)..."
    )

    with file_path.open(
        "rb"
    ) as file_handle:

        encoded = base64.b64encode(
            file_handle.read()
        ).decode(
            "utf-8"
        )


    # -------------------------------------------------------------------------
    # Request payload
    # -------------------------------------------------------------------------

    payload = {
        "type": "file",
        "folderName": folder_name,
        "filename": file_path.name,
        "mimeType": mime_type,
        "base64": encoded,
    }

    if message:
        payload["message"] = message


    # -------------------------------------------------------------------------
    # POST to Apps Script
    # -------------------------------------------------------------------------

    response = requests.post(
        UPLOAD_URL,
        json=payload,
        timeout=UPLOAD_TIMEOUT_SECONDS,
    )

    response.raise_for_status()


    # -------------------------------------------------------------------------
    # Parse response
    # -------------------------------------------------------------------------

    try:
        result = response.json()

    except ValueError as error:

        raise RuntimeError(
            "Apps Script returned an invalid JSON response:\n"
            f"{response.text}"
        ) from error


    if not result.get(
        "success"
    ):

        raise RuntimeError(
            result.get(
                "error",
                "Unknown Google Drive upload error",
            )
        )

    return result


# =============================================================================
# ZIP + upload
# =============================================================================

def upload_zip(
    identifier: str,
    drive_folder: str = DEFAULT_DRIVE_FOLDER,
    message: str | None = None,
) -> dict:
    """
    Zip an experiment directory and upload the ZIP to Google Drive.

    Args:
        identifier:
            Experiment directory.

        drive_folder:
            Destination folder name on Google Drive.

        message:
            Optional custom message included in the
            notification email.

    Returns:
        JSON response returned by Apps Script.
    """

    output_dir = Path(
        identifier
    )

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist: "
            f"{output_dir}"
        )


    # -------------------------------------------------------------------------
    # Package experiment
    # -------------------------------------------------------------------------

    zip_path = create_zip(
        output_dir
    )


    # -------------------------------------------------------------------------
    # Upload
    # -------------------------------------------------------------------------

    print(
        f"[Upload] Uploading '{zip_path}' "
        f"to Google Drive folder "
        f"'{drive_folder}'..."
    )

    result = upload_file(
        zip_path,
        drive_folder,
        message=message,
    )


    # -------------------------------------------------------------------------
    # Report result
    # -------------------------------------------------------------------------

    uploaded_name = result.get("filename") or zip_path.name
    print(
        f"[Upload] ✓ Upload successful — "
        f"{uploaded_name}"
    )

    if "url" in result:

        print(
            f"[Upload] URL: "
            f"{result['url']}"
        )


    notification_sent = result.get(
        "notificationSent"
    )

    if notification_sent is True:

        print(
            "[Upload] ✓ Email notification sent"
        )

    elif notification_sent is False:

        print(
            "[Upload] ⚠ File was uploaded successfully, "
            "but the email notification failed"
        )

    else:

        print(
            "[Upload] ⚠ Server did not report "
            "email notification status"
        )


    print(
        f"[Registration] ✓ Experiment "
        f"'{identifier}' registered successfully."
    )

    return result
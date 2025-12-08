# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "paramiko",
#     "python-dotenv",
#     "tqdm",
# ]
# ///

# run with: uv run --env-file .env.local .\data\download.py

import os

from tqdm import tqdm  # type: ignore
from dotenv import load_dotenv  # type: ignore
import paramiko  # type: ignore
from pathlib import Path

# Load sensitive info
load_dotenv()
REMOTE_USER = os.getenv("REMOTE_USER")
if not REMOTE_USER:
    raise ValueError("REMOTE_USER not set in environment variables")

REMOTE_HOST = os.getenv("REMOTE_HOST")
if not REMOTE_HOST:
    raise ValueError("REMOTE_HOST not set in environment variables")

REMOTE_PORT_STR = os.getenv("REMOTE_PORT")
if not REMOTE_PORT_STR:
    raise ValueError("REMOTE_PORT not set in environment variables")
REMOTE_PORT = int(REMOTE_PORT_STR)

REMOTE_PATH_STR = os.getenv("REMOTE_PATH")
if not REMOTE_PATH_STR:
    raise ValueError("REMOTE_PATH not set in environment variables")
REMOTE_PATH = Path(REMOTE_PATH_STR)

PASSWORD = os.getenv("PASSWORD")  # optional if using key-based auth
if not PASSWORD:
    raise ValueError("PASSWORD not set in environment variables")

LOCAL_PATH = Path(__file__).parent.resolve() / "flux" / "raw"
REMOTE_TARGET_FILE_NAME: str = "fluxes.dat"
LOCAL_FILE_NAME_CONVENTION: str = "fluxes_{iteration}.dat"

# Ensure local folder exists
os.makedirs(LOCAL_PATH, exist_ok=True)

# Create SSH client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    REMOTE_HOST,
    port=REMOTE_PORT,
    username=REMOTE_USER,
    password=PASSWORD,
)

# Open SFTP session
sftp = ssh.open_sftp()


# List iteration folders
folders = [
    f
    for f in sftp.listdir(REMOTE_PATH.as_posix())
    if f.startswith("iteration_") and not f.endswith("Lin")
]

for folder in tqdm(folders):
    iteration = folder.split("_")[-1]
    remote_folder: Path = REMOTE_PATH / folder
    remote_file = remote_folder / REMOTE_TARGET_FILE_NAME
    local_file = LOCAL_PATH / LOCAL_FILE_NAME_CONVENTION.format(iteration=iteration)
    try:
        tqdm.write(f"Downloading {remote_file} -> {local_file}")
        sftp.get(remote_file.as_posix(), local_file.as_posix())
    except IOError:
        tqdm.write(f"File {remote_file} does not exist, skipping.")

# Close connections
sftp.close()
ssh.close()
print("Download complete!")

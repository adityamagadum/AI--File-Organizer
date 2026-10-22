from main import image_inference, text_inference, initialize_models


import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# import the main module to access initialize_models and model objects
import main as main_mod

from file_utils import separate_files_by_type, read_file_data
from data_processing_common import (
    compute_operations,
    execute_operations,
    process_files_by_type,
    process_files_by_date,
)
from image_data_processing import process_image_files
from text_data_processing import process_text_files

# -------------------------
# CONFIG — edit these paths (use raw strings r"...")
# -------------------------
INPUT_FOLDER = r"C:\\Users\\Abhishek\\Downloads\\Telegram Desktop\\New folder"
OUTPUT_FOLDER = r"C:\\Users\\Abhishek\\Downloads\\Telegram Desktop\\New folder\\organized_folder"
MODE = "type"     # choose: "type" | "date" | "content"
# -------------------------

_FILE_STABLE_WAIT = 0.6
_FILE_STABLE_CHECKS = 2
_FILE_READY_TIMEOUT = 60  # seconds

def wait_for_file_ready(path, timeout=_FILE_READY_TIMEOUT):
    """Wait until the file size stabilizes (to avoid processing partial copies)."""
    start = time.time()
    prev_size = -1
    stable_count = 0
    while True:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size == prev_size and size > 0:
            stable_count += 1
        else:
            stable_count = 0
        prev_size = size
        if stable_count >= _FILE_STABLE_CHECKS:
            return True
        if time.time() - start > timeout:
            return False
        time.sleep(_FILE_STABLE_WAIT)

class NewFileHandler(FileSystemEventHandler):
    """Handler to process newly created files."""

    def on_created(self, event):
        if event.is_directory:
            return

        new_file = event.src_path
        print(f"\n⚡ New file detected: {new_file}")

        ready = wait_for_file_ready(new_file)
        if not ready:
            print(f"⚠️ Timeout waiting for file to be ready: {new_file}. Skipping.")
            return

        time.sleep(0.2)

        try:
            if MODE == "type":
                self.process_type_mode(new_file)
            elif MODE == "date":
                self.process_date_mode(new_file)
            elif MODE == "content":
                self.process_content_mode(new_file)
            else:
                print(f"Unknown MODE: {MODE} — no action taken.")
        except Exception as e:
            print(f"Error while processing {new_file}: {e}")

    def process_type_mode(self, file_path):
        print("Sorting by TYPE...")
        ops = process_files_by_type([file_path], OUTPUT_FOLDER)
        execute_operations(ops, dry_run=False)

        # DELETE ORIGINAL FILE
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted original file: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not delete original file: {e}")

        print("✔ File sorted by TYPE")

    def process_date_mode(self, file_path):
        print("Sorting by DATE...")
        ops = process_files_by_date([file_path], OUTPUT_FOLDER)
        execute_operations(ops, dry_run=False)

        # DELETE ORIGINAL FILE
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted original file: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not delete original file: {e}")

        print("✔ File sorted by DATE")

    def process_content_mode(self, file_path):
        print("Sorting by CONTENT (AI)...")

        # Initialize models if needed
        if getattr(main_mod, "image_inference", None) is None or getattr(main_mod, "text_inference", None) is None:
            print("Initializing AI models (this may take a while on first run)...")
            try:
                main_mod.initialize_models()
            except Exception as e:
                print(f"⚠️ Error initializing models: {e}")
                self.process_type_mode(file_path)
                return

            wait_start = time.time()
            while (main_mod.image_inference is None or main_mod.text_inference is None) and time.time() - wait_start < 120:
                time.sleep(0.5)
            if main_mod.image_inference is None or main_mod.text_inference is None:
                print("⚠️ Models didn't initialize in time. Falling back to type sort.")
                self.process_type_mode(file_path)
                return

        image_files, text_files = separate_files_by_type([file_path])
        data_list = []

        if file_path in image_files:
            data_list += process_image_files([file_path], main_mod.image_inference, main_mod.text_inference)

        if file_path in text_files:
            text_content = read_file_data(file_path)
            if text_content is None:
                print(f"⚠️ Could not read content of {file_path}.")
            else:
                data_list += process_text_files([(file_path, text_content)], main_mod.text_inference)

        if not data_list:
            print("No AI metadata produced for this file. Falling back to TYPE sort.")
            self.process_type_mode(file_path)
            return

        ops = compute_operations(data_list, OUTPUT_FOLDER, set(), set())
        execute_operations(ops, dry_run=False)

        # DELETE ORIGINAL FILE
        try:
            os.remove(file_path)
            print(f"🗑️ Deleted original file: {file_path}")
        except Exception as e:
            print(f"⚠️ Could not delete original file: {e}")

        print("✔ File sorted by CONTENT")

def start_watching():
    if not os.path.isdir(INPUT_FOLDER):
        raise SystemExit(f"Input folder does not exist: {INPUT_FOLDER}")

    print("\n🔍 Watching for new files...")
    print(f"Input Folder: {INPUT_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print(f"Mode: {MODE}")
    print("Press CTRL + C to stop.\n")

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUT_FOLDER, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("Watcher stopped by user.")
    observer.join()

if __name__ == "__main__":
    start_watching()

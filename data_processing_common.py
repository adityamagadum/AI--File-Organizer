# data_processing_common.py
import os
import re
import datetime
import shutil
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn

def sanitize_filename(name, max_length=50, max_words=5):
    """Sanitize the filename by removing unwanted words and characters."""
    name = os.path.splitext(name)[0]
    name = re.sub(
        r'\b(jpg|jpeg|png|gif|bmp|txt|md|pdf|docx|xls|xlsx|csv|ppt|pptx|image|picture|photo|this|that|these|those|here|there|'
        r'please|note|additional|notes|folder|name|sure|heres|a|an|the|and|of|in|'
        r'to|for|on|with|your|answer|should|be|only|summary|summarize|text|category)\b',
        '',
        name,
        flags=re.IGNORECASE
    )
    sanitized = re.sub(r'[^\w\s]', '', name).strip()
    sanitized = re.sub(r'[\s_]+', '_', sanitized)
    sanitized = sanitized.lower().strip('_')
    words = sanitized.split('_')
    limited_words = [word for word in words if word]
    limited_words = limited_words[:max_words]
    limited_name = '_'.join(limited_words)
    return limited_name[:max_length] if limited_name else 'untitled'

def process_files_by_date(file_paths, output_path, dry_run=False, silent=False, log_file=None):
    """Process files to organize them by date."""
    operations = []
    for file_path in file_paths:
        mod_time = os.path.getmtime(file_path)
        mod_datetime = datetime.datetime.fromtimestamp(mod_time)
        year = mod_datetime.strftime('%Y')
        month = mod_datetime.strftime('%B')
        dir_path = os.path.join(output_path, year, month)
        new_file_name = os.path.basename(file_path)
        new_file_path = os.path.join(dir_path, new_file_name)
        link_type = 'hardlink'
        operation = {
            'source': file_path,
            'destination': new_file_path,
            'link_type': link_type,
        }
        operations.append(operation)
    return operations

def process_files_by_type(file_paths, output_path, dry_run=False, silent=False, log_file=None):
    """Process files to organize them by type, first separating into text-based and image-based files."""
    operations = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
    text_extensions = ('.txt', '.md', '.docx', '.doc', '.pdf', '.xls', '.xlsx', '.epub', '.mobi', '.azw', '.azw3')

    for file_path in file_paths:
        if os.path.basename(file_path).startswith('.'):
            continue
        ext = os.path.splitext(file_path)[1].lower()
        if ext in image_extensions:
            top_folder = 'image_files'
            folder_name = top_folder
        elif ext in text_extensions:
            top_folder = 'text_files'
            if ext in ('.txt', '.md'):
                sub_folder = 'plain_text_files'
            elif ext in ('.doc', '.docx'):
                sub_folder = 'doc_files'
            elif ext == '.pdf':
                sub_folder = 'pdf_files'
            elif ext in ('.xls', '.xlsx'):
                sub_folder = 'xls_files'
            elif ext in ('.epub', '.mobi', '.azw', '.azw3'):
                sub_folder = 'ebooks'
            else:
                sub_folder = 'others'
            folder_name = os.path.join(top_folder, sub_folder)
        else:
            folder_name = 'others'
        dir_path = os.path.join(output_path, folder_name)
        new_file_name = os.path.basename(file_path)
        new_file_path = os.path.join(dir_path, new_file_name)
        link_type = 'hardlink'
        operation = {
            'source': file_path,
            'destination': new_file_path,
            'link_type': link_type,
        }
        operations.append(operation)
    return operations

def compute_operations(data_list, new_path, renamed_files, processed_files):
    """Compute the file operations based on generated metadata."""
    operations = []
    for data in data_list:
        file_path = data['file_path']
        if file_path in processed_files:
            continue
        processed_files.add(file_path)

        folder_name = data['foldername']
        new_file_name = data['filename'] + os.path.splitext(file_path)[1]
        dir_path = os.path.join(new_path, folder_name)
        new_file_path = os.path.join(dir_path, new_file_name)

        counter = 1
        original_new_file_name = new_file_name
        while new_file_path in renamed_files:
            new_file_name = f"{data['filename']}_{counter}" + os.path.splitext(file_path)[1]
            new_file_path = os.path.join(dir_path, new_file_name)
            counter += 1

        link_type = 'hardlink'
        operation = {
            'source': file_path,
            'destination': new_file_path,
            'link_type': link_type,
            'folder_name': folder_name,
            'new_file_name': new_file_name
        }
        operations.append(operation)
        renamed_files.add(new_file_path)

    return operations

def execute_operations(operations, dry_run=False, silent=False, log_file=None):
    """Execute the file operations. Tries hardlink, falls back to copy."""
    total_operations = len(operations)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True
    ) as progress:
        task = progress.add_task("Organizing Files...", total=total_operations)
        for operation in operations:
            source = operation['source']
            destination = operation['destination']
            link_type = operation.get('link_type', 'hardlink')
            dir_path = os.path.dirname(destination)

            if dry_run:
                message = f"Dry run: would create {link_type} from '{source}' to '{destination}'"
            else:
                os.makedirs(dir_path, exist_ok=True)
                try:
                    # Try creating a hardlink first
                    os.link(source, destination)
                    message = f"Created hardlink from '{source}' to '{destination}'"
                except Exception as e_link:
                    try:
                        # Fallback: copy file (preserve metadata)
                        shutil.copy2(source, destination)
                        message = f"Copied file from '{source}' to '{destination}' (fallback after hardlink failed: {e_link})"
                    except Exception as e_copy:
                        message = f"Error moving/copying from '{source}' to '{destination}': {e_copy}"

            progress.advance(task)

            if silent:
                if log_file:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(message + '\n')
            else:
                print(message)

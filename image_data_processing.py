import re
import os
import time
from typing import Any, Iterator, Optional

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn
from data_processing_common import sanitize_filename  # Import sanitize_filename


def get_text_from_generator(generator: Any) -> str:
    """
    Accept and normalise different model outputs to a plain string.

    Supported inputs:
      - plain str -> returned as-is
      - iterator/generator yielding str chunks -> concatenated
      - dict with 'choices' -> concatenates choice.delta.content or choice['text']
      - list/dict with 'text' fields -> concatenated
    """
    if generator is None:
        return ""

    # Plain string -> return immediately
    if isinstance(generator, str):
        return generator

    # If it's a dict with 'choices' or 'text'
    if isinstance(generator, dict):
        # streaming-like dict with choices
        if 'choices' in generator:
            parts = []
            for choice in generator['choices']:
                # streaming delta style
                delta = choice.get('delta', {})
                if isinstance(delta, dict) and 'content' in delta:
                    parts.append(delta['content'])
                elif 'text' in choice:
                    parts.append(choice.get('text', ''))
            return "".join(parts)
        # single text field
        if 'text' in generator:
            return str(generator['text'])

    # If it's a list of items
    if isinstance(generator, list):
        parts = []
        for item in generator:
            parts.append(get_text_from_generator(item))
        return "".join(parts)

    # If it's an iterator/generator, consume it safely
    try:
        iterator = iter(generator)
    except TypeError:
        # fallback to string conversion
        return str(generator)

    parts = []
    for item in iterator:
        parts.append(get_text_from_generator(item))
    return "".join(parts)


def _safe_model_call(model_obj: Any, method_name: str, *args, **kwargs) -> Any:
    """
    Safely call a model method and return its result.
    Catches exceptions and returns an empty string on failure.
    """
    try:
        method = getattr(model_obj, method_name)
        result = method(*args, **kwargs)
        return result
    except Exception:
        # As a last resort, some models may implement create_completion as a callable attribute
        try:
            if callable(model_obj):
                return model_obj(*args, **kwargs)
        except Exception:
            pass
    return ""


def process_single_image(image_path: str, image_inference: Any, text_inference: Any,
                         silent: bool = False, log_file: Optional[str] = None) -> dict:
    """Process a single image file to generate metadata."""
    start_time = time.time()

    # Create a Progress instance for this file
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as progress:
        task_id = progress.add_task(f"Processing {os.path.basename(image_path)}", total=1.0)
        foldername, filename, description = generate_image_metadata(
            image_path, progress, task_id, image_inference, text_inference
        )

    end_time = time.time()
    time_taken = end_time - start_time

    message = (
        f"File: {image_path}\nTime taken: {time_taken:.2f} seconds\n"
        f"Description: {description}\nFolder name: {foldername}\nGenerated filename: {filename}\n"
    )
    if silent and log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    elif not silent:
        print(message)

    return {
        'file_path': image_path,
        'foldername': foldername,
        'filename': filename,
        'description': description
    }


def process_image_files(image_paths: list, image_inference: Any, text_inference: Any,
                        silent: bool = False, log_file: Optional[str] = None) -> list:
    """Process image files sequentially."""
    data_list = []
    for image_path in image_paths:
        data = process_single_image(image_path, image_inference, text_inference,
                                    silent=silent, log_file=log_file)
        data_list.append(data)
    return data_list


def generate_image_metadata(image_path: str, progress, task_id: int,
                            image_inference: Any, text_inference: Any):
    """Generate description, folder name, and filename for an image file."""

    total_steps = 3
    # Step 1: Generate description using image_inference
    description_prompt = (
        "Please provide a detailed description of this image, focusing on the main subject and any important details."
    )

    raw_description = _safe_model_call(image_inference, "_chat", description_prompt, image_path)
    # Normalise to text (handles strings, iterators, dicts)
    description = get_text_from_generator(raw_description).strip()
    if not description:
        # fallback short message
        description = f"Description not available for {os.path.basename(image_path)}"
    progress.update(task_id, advance=1 / total_steps)

    # --- Helper to extract text from the text_inference responses
    def extract_completion_text(resp: Any) -> str:
        text = get_text_from_generator(resp).strip()
        # If resp is dict with choices and text inside choices
        if not text and isinstance(resp, dict) and 'choices' in resp:
            parts = []
            for c in resp.get('choices', []):
                if isinstance(c, dict):
                    if 'text' in c:
                        parts.append(c['text'])
                    elif 'delta' in c and 'content' in c['delta']:
                        parts.append(c['delta']['content'])
            text = "".join(parts).strip()
        return text

    # Step 2: Generate filename using text_inference
    filename_prompt = f"""Based on the description below, generate a specific and descriptive filename for the image.
Limit the filename to a maximum of 3 words. Use nouns and avoid starting with verbs like 'depicts', 'shows', 'presents', etc.
Do not include any data type words like 'image', 'jpg', 'png', etc. Use only letters and connect words with underscores.

Description: {description}

Now generate the filename.

Output only the filename, without any additional text.

Filename:"""

    # Try common method names for completions (create_completion, create, completion, or _chat)
    raw_filename_response = (
        _safe_model_call(text_inference, "create_completion", filename_prompt)
        or _safe_model_call(text_inference, "create", filename_prompt)
        or _safe_model_call(text_inference, "create_chat_completion", filename_prompt)
        or _safe_model_call(text_inference, "_chat", filename_prompt)
        or _safe_model_call(text_inference, "completion", filename_prompt)
    )
    filename_text = extract_completion_text(raw_filename_response)
    filename_text = re.sub(r'^Filename:\s*', '', filename_text, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    # Step 3: Generate folder name from description using text_inference
    foldername_prompt = f"""Based on the description below, generate a general category or theme that best represents the main subject of this image.
This will be used as the folder name. Limit the category to a maximum of 2 words. Use nouns and avoid verbs.

Description: {description}

Output only the category, without any additional text.

Category:"""

    raw_folder_response = (
        _safe_model_call(text_inference, "create_completion", foldername_prompt)
        or _safe_model_call(text_inference, "create", foldername_prompt)
        or _safe_model_call(text_inference, "create_chat_completion", foldername_prompt)
        or _safe_model_call(text_inference, "_chat", foldername_prompt)
        or _safe_model_call(text_inference, "completion", foldername_prompt)
    )
    folder_text = extract_completion_text(raw_folder_response)
    folder_text = re.sub(r'^Category:\s*', '', folder_text, flags=re.IGNORECASE).strip()
    progress.update(task_id, advance=1 / total_steps)

    # Remove any unwanted words and stopwords
    unwanted_words = set([
        'the', 'and', 'based', 'generated', 'this', 'is', 'filename', 'file', 'image', 'picture', 'photo',
        'folder', 'category', 'output', 'only', 'below', 'text', 'jpg', 'png', 'jpeg', 'gif', 'bmp', 'svg',
        'logo', 'in', 'on', 'of', 'with', 'by', 'for', 'to', 'from', 'a', 'an', 'as', 'at', 'red', 'blue',
        'green', 'color', 'colors', 'colored', 'graphic', 'graphics', 'main', 'subject', 'important',
        'details', 'description', 'depicts', 'show', 'shows', 'display', 'illustrates', 'presents', 'features',
        'provides', 'covers', 'includes', 'demonstrates', 'describes'
    ])
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()
    all_unwanted_words = unwanted_words.union(stop_words)
    lemmatizer = WordNetLemmatizer()

    def clean_ai_output(text: str, max_words: int) -> str:
        if not text:
            return ""
        # Remove file extensions and reserved characters
        text = re.sub(r'\.\w{1,4}$', '', text)  # trailing extension
        text = re.sub(r'[^\w\s]', ' ', text)  # punctuation -> space
        text = re.sub(r'\d+', ' ', text)  # remove digits
        text = text.strip()
        # Split concatenated camelCase words
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        # Tokenize and lemmatize words
        try:
            words = word_tokenize(text)
        except Exception:
            # fallback simple split
            words = text.split()
        words = [w.lower() for w in words if w.isalpha()]
        words = [lemmatizer.lemmatize(w) for w in words]
        # Remove unwanted words and duplicates while preserving order
        filtered = []
        seen = set()
        for w in words:
            if w not in all_unwanted_words and w not in seen:
                filtered.append(w)
                seen.add(w)
        filtered = filtered[:max_words]
        return '_'.join(filtered)

    # Process filename: prefer filename_text, fallback to description
    filename = clean_ai_output(filename_text, max_words=3)
    if not filename:
        filename = clean_ai_output(description, max_words=3)
    if not filename:
        base = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"image_{re.sub(r'[^\w]', '_', base)}"

    sanitized_filename = sanitize_filename(filename, max_words=3)

    # Process foldername: prefer folder_text, fallback to description, then default 'images'
    foldername = clean_ai_output(folder_text, max_words=2)
    if not foldername:
        foldername = clean_ai_output(description, max_words=2)
    if not foldername:
        foldername = 'images'

    sanitized_foldername = sanitize_filename(foldername, max_words=2)

    return sanitized_foldername, sanitized_filename, description

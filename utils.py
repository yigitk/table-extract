import logging
import pandas as pd
import re
from pathlib import Path
import csv
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
import fitz  # PyMuPDF
import dateparser # Add this import if you want to parse/validate the date string
from datetime import date as date_type # Make sure date is imported

# --- Module Level Constants and Compiled Regex ---

# Ensure _log is defined early if needed for InputFormat fallback
_log = logging.getLogger(__name__)

# Try importing InputFormat (used in _create_document_converter)
try:
    from docling.datamodel.base_models import InputFormat
except ImportError:
    _log.warning("Could not import InputFormat from docling.datamodel.base_models. InputFormat type hints may be incorrect.")
    InputFormat = Any # Fallback if InputFormat also moved

# Moved unit_pattern definition to module level for importability
unit_pattern_str = r"""
    # Specific volume/concentration (longer first)
    ng/L|mg/g\b|ug/dL|mg/dL|u/L|mg/L|ug/L|ng/mL|ug/mL|g/dL|pg/mL|
    nmol/L|mg/mmol|umol/L|mmol/L|nmol/mL|pmol/L|
    # Molar/Activity
    mosm/KG|uIU/L|uIU/mL|IU/L|IU/mL|mIU/mL|arb\'U/mL|
    # Cells/Counts (careful with boundaries)
    1000\s*cells/ul|cells/ul|/ul|10\*3/ul|x10\^12/L|10\*6/mL|count\b|
    # Mass/Dimensions/Physical (longer first)
    g/cm2|ug/g\[Hb\]|g/g\{creat\}|mg/mL|mm\[HG\]|mmHg|mL/min/1\.73\s*m[2²]|
    KG|%|Ω|L\b|fL|mL\b|mm/h|mm/hr|cm\b|kPa|ratio|
    # Other common lab terms
    ph\b|clarity|color|pattern|score|type\b|present|positive|negative|detected|not detected|normal|abnormal|reactive|non-reactive|high|low|
    # Time/Rate
    s\b|sec|min|hr|kcal/kg/h|
    # Wavelength/Misc
    nm\b
"""
# Create compact string first
compact_unit_pattern_str = unit_pattern_str.replace(chr(10), "").replace(" ", "")
# Compile the compact string with the outer capturing group and IGNORECASE
unit_pattern = re.compile(f"({compact_unit_pattern_str})", re.IGNORECASE)

# --- End Module Level Definitions ---


from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    # EasyOcrOptions,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.settings import settings

# --- Biomarker Matching Constants ---
KNOWN_BIOMARKERS_FILE = Path("biomarkers.csv") # Assuming file is in the root
KNOWN_BIOMARKERS_LOOKUP: Dict[str, str] = {} # Cache for loaded biomarkers
# Minimum Jaccard score for a partial match to be considered valid
MIN_JACCARD_THRESHOLD = 0.5

# --- Configuration Functions ---
# (Keep _configure_rapid_ocr, _create_document_converter as they are backend logic)
def _configure_rapid_ocr() -> PdfPipelineOptions:
    """Configure RapidOCR pipeline options."""
    accelerator_options = AcceleratorOptions(
        num_threads=4,
        device=AcceleratorDevice.CPU
    )
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_page_images = False
    pipeline_options.ocr_options = RapidOcrOptions(
        force_full_page_ocr=True,
        use_cls=True,
        text_score=0.7
    )
    return pipeline_options

def _create_document_converter(pipeline_options: PdfPipelineOptions) -> DocumentConverter:
    """Create document converter with given pipeline options."""
    # Assuming InputFormat was successfully imported or is Any
    allowed_formats = [InputFormat.PDF, InputFormat.IMAGE, InputFormat.DOCX] if InputFormat is not Any else ["pdf", "png", "jpg", "jpeg", "tiff", "docx"] # Fallback if type is lost

    return DocumentConverter(
        allowed_formats=allowed_formats,
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.DOCX: PdfFormatOption(pipeline_options=pipeline_options),
        } if InputFormat is not Any else { # Fallback if type is lost
            "pdf": PdfFormatOption(pipeline_options=pipeline_options),
            "png": PdfFormatOption(pipeline_options=pipeline_options),
            "jpg": PdfFormatOption(pipeline_options=pipeline_options),
            "jpeg": PdfFormatOption(pipeline_options=pipeline_options),
            "tiff": PdfFormatOption(pipeline_options=pipeline_options),
            "docx": PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

# --- Biomarker Matching Utilities ---

def _clean_biomarker_name(name: str) -> str:
    """Standardizes a biomarker name for matching."""
    if not isinstance(name, str):
        return ""
    # Lowercase, remove specific punctuation potentially separating parts, strip whitespace
    cleaned = name.lower()
    cleaned = re.sub(r'[(),-/"]', ' ', cleaned) # Replace common separators with space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip() # Consolidate whitespace and strip ends
    return cleaned

def load_known_biomarkers(
    file_path: Path = KNOWN_BIOMARKERS_FILE,
    force_reload: bool = False
) -> Dict[str, str]:
    """
    Loads known biomarkers from a CSV file into a lookup dictionary.

    The CSV should have 'Biomarker | Interventions' and 'Biomarker ID' columns.
    It cleans the biomarker names for robust matching.

    Args:
        file_path: The path to the CSV file.
        force_reload: If True, forces reloading even if cache exists.

    Returns:
        A dictionary mapping cleaned biomarker names to their IDs.
        Logs an error and returns an empty dict if loading fails.
    """
    global KNOWN_BIOMARKERS_LOOKUP
    if KNOWN_BIOMARKERS_LOOKUP and not force_reload:
        _log.info("Using cached known biomarkers.")
        return KNOWN_BIOMARKERS_LOOKUP

    _log.info(f"Loading known biomarkers from: {file_path}")
    biomarker_lookup: Dict[str, str] = {}
    try:
        if not file_path.is_file():
            _log.error(f"Known biomarkers file not found at: {file_path}")
            return {}

        with open(file_path, mode='r', encoding='utf-8') as infile:
            # Handle potential variations in header names slightly more robustly
            reader = csv.DictReader(infile)
            # Dynamically find header names (case-insensitive, ignoring surrounding spaces/pipes)
            headers = [h.strip().strip('|').strip().lower() for h in reader.fieldnames or []]
            name_col = next((h for h in reader.fieldnames or [] if 'biomarker | interventions' in h.strip().strip('|').strip().lower()), None)
            id_col = next((h for h in reader.fieldnames or [] if 'biomarker id' in h.strip().strip('|').strip().lower()), None)

            if not name_col or not id_col:
                 _log.error(f"Could not find required columns ('Biomarker | Interventions', 'Biomarker ID') in {file_path}. Found headers: {reader.fieldnames}")
                 return {}

            _log.debug(f"Using Name Column: '{name_col}', ID Column: '{id_col}'")

            for row in reader:
                biomarker_name = row.get(name_col)
                biomarker_id = row.get(id_col)
                if biomarker_name and biomarker_id:
                    cleaned_name = _clean_biomarker_name(biomarker_name)
                    if cleaned_name:
                        # Handle potential duplicates in cleaned names - keeping the first ID encountered
                        if cleaned_name not in biomarker_lookup:
                            biomarker_lookup[cleaned_name] = biomarker_id.strip()
                        else:
                             _log.warning(f"Duplicate cleaned biomarker name '{cleaned_name}' found. Keeping first ID '{biomarker_lookup[cleaned_name]}', ignoring new ID '{biomarker_id.strip()}'. Original name: '{biomarker_name}'")
        _log.info(f"Successfully loaded {len(biomarker_lookup)} unique known biomarkers.")
        KNOWN_BIOMARKERS_LOOKUP = biomarker_lookup # Update cache
        return biomarker_lookup

    except FileNotFoundError:
        _log.error(f"Known biomarkers file not found at: {file_path}")
        KNOWN_BIOMARKERS_LOOKUP = {}
        return {}
    except Exception as e:
        _log.exception(f"Error loading or processing known biomarkers file {file_path}: {e}")
        KNOWN_BIOMARKERS_LOOKUP = {}
        return {}


def match_biomarker(
    raw_name: str,
    known_biomarkers: Dict[str, str]
) -> Tuple[Optional[str], bool]:
    """
    Matches a raw biomarker name against a dictionary of known biomarkers.

    Performs cleaning, checks for exact matches, and then uses Jaccard Index
    based on word overlap for partial matching.

    Args:
        raw_name: The raw biomarker name extracted from the document.
        known_biomarkers: The dictionary loaded by load_known_biomarkers.

    Returns:
        A tuple: (matched_biomarker_id, matched_flag).
        matched_flag is True if an exact or partial match (above threshold) is found.
        matched_biomarker_id is the ID if matched, None otherwise.
    """
    if not raw_name or not isinstance(raw_name, str) or not known_biomarkers:
        return None, False

    cleaned_raw_name = _clean_biomarker_name(raw_name)
    if not cleaned_raw_name:
        return None, False

    # 1. Exact Match (based on cleaned names)
    if cleaned_raw_name in known_biomarkers:
        _log.debug(f"Exact match found for '{raw_name}' (cleaned: '{cleaned_raw_name}') -> ID: {known_biomarkers[cleaned_raw_name]}")
        return known_biomarkers[cleaned_raw_name], True

    # 2. Partial Match using Jaccard Index on Word Sets
    _log.debug(f"No exact match for '{raw_name}' (cleaned: '{cleaned_raw_name}'). Checking partial matches...")
    raw_words = set(cleaned_raw_name.split())
    if not raw_words: # Cannot match if raw name becomes empty after cleaning/splitting
         return None, False

    best_match_id: Optional[str] = None
    best_match_score: float = -1.0 # Initialize below threshold

    for known_name_cleaned, known_id in known_biomarkers.items():
        known_words = set(known_name_cleaned.split())
        if not known_words: continue # Skip empty known names

        # Calculate Jaccard Index
        intersection = raw_words.intersection(known_words)
        union = raw_words.union(known_words)
        
        if not union: continue # Avoid division by zero if both are empty somehow

        jaccard_score = len(intersection) / len(union)

        # Check if this score is the best so far AND meets the threshold
        if jaccard_score > best_match_score and jaccard_score >= MIN_JACCARD_THRESHOLD:
            best_match_score = jaccard_score
            best_match_id = known_id
            _log.debug(f"  Potential partial match: '{raw_name}' ~ '{known_name_cleaned}' (Score: {jaccard_score:.3f}) -> ID: {known_id}")

    # 3. Evaluate Best Partial Match Found
    if best_match_id is not None:
        _log.info(f"Best partial match for '{raw_name}' found with score {best_match_score:.3f} -> ID: {best_match_id}")
        return best_match_id, True
    else:
        _log.debug(f"No partial match found above threshold {MIN_JACCARD_THRESHOLD} for '{raw_name}'")
        return None, False

# --- API Specific Processing Function ---
async def process_document_for_api(file_path: Path) -> Optional[Any]:
    """
    Processes a document file using RapidOCR for API usage.

    Args:
        file_path: Path to the document file to process.

    Returns:
        The conversion result object from docling, or None if processing fails.
        Type hint uses Any due to import issues with ConversionResult.
    """
    rapid_pipeline_options = _configure_rapid_ocr()
    doc_converter_rapid = _create_document_converter(rapid_pipeline_options)
    settings.debug.profile_pipeline_timings = False # Ensure timing is off for API

    try:
        _log.info(f"Processing document for API with RapidOCR: {file_path}")
        # Assuming convert is blocking or handles its own async internally
        rapid_conv_res: Optional[Any] = doc_converter_rapid.convert(file_path)
        _log.info(f"Document processing completed for: {file_path}")
        return rapid_conv_res
    except Exception as e:
        _log.exception(f"Error during docling conversion for {file_path}: {e}")
        return None

# --- File Preview Function (Removed Streamlit dependency) ---
# This is kept as it might be useful backend logic, but doesn't use streamlit
# Returns image object and error string
def preview_pdf_fitz(pdf_bytes: bytes, page: int = 1) -> Tuple[Optional[Image.Image], Optional[str]]:
    """PDF preview function using PyMuPDF (Backend version)."""
    pdf_document = None
    try:
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if not pdf_document or page < 1 or page > len(pdf_document):
            _log.warning(f"Invalid page number {page} requested for PDF.")
            return None, f"Invalid page number {page} for PDF with {len(pdf_document)} pages."

        page_obj = pdf_document.load_page(page - 1)
        pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2)) # zoom=2x
        if pix.samples is None:
             _log.error("Failed to get pixmap samples.")
             return None, "Failed to render PDF page."
        # Create PIL Image
        img: Image.Image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img, None
    except Exception as e:
        _log.exception(f"Error previewing PDF page {page}: {str(e)}")
        return None, f"Error previewing PDF: {str(e)}"
    finally:
        if pdf_document:
            try:
                pdf_document.close()
            except Exception as e:
                _log.error(f"Error closing PDF document: {e}")

# --- REMOVED Streamlit UI Functions ---
# Removed: show_file_preview(uploaded_file)
# Removed: process_tables_and_biomarkers(...)
# Removed: process_document_for_streamlit(...) - API function replaces this need

# --- Table Processing Functions ---
# (Keep clean_column_name, rename_duplicate_columns, merge_duplicate_columns,
#  standardize_column_names, clean_table_data as they are general utilities)
def clean_column_name(col_name: str) -> str:
    """Clean individual column name (without handling duplicates)."""
    col = str(col_name).strip('| ').strip()
    col = ' '.join(col.split()) # Consolidate whitespace
    return col

def rename_duplicate_columns(cols: List) -> List[str]:
    """Rename duplicate columns by appending a suffix."""
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for col in cols:
        cleaned = clean_column_name(str(col))
        if not cleaned:
             cleaned = f"Unnamed_{len(new_cols)}" # Handle empty names
        if cleaned in seen:
            seen[cleaned] += 1
            new_cols.append(f"{cleaned}_{seen[cleaned]}")
        else:
            seen[cleaned] = 0
            new_cols.append(cleaned)
    return new_cols

def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge columns that originally had the same header by concatenating their values row-wise."""
    if df.empty:
        return pd.DataFrame()

    original_names = [clean_column_name(str(col)) for col in df.columns]

    new_df = pd.DataFrame(index=df.index) # Preserve original index
    unique_cols = list(dict.fromkeys(name for name in original_names if name)) # Filter out empty names
    
    for col_name in unique_cols:
        indices = [i for i, name in enumerate(original_names) if name == col_name]

        if len(indices) == 1:
            # Copy single column directly
            new_df[col_name] = df.iloc[:, indices[0]]
        elif len(indices) > 1:
            # Merge multiple columns, ensuring string conversion and handling NaNs
            combined_series = df.iloc[:, indices].apply(
                lambda row: ' '.join(
                    str(x) for x in row if pd.notna(x) and str(x).strip()
                ).strip(),
                axis=1
            )
            # Only add column if it contains non-empty data after merging
            if combined_series.astype(bool).any(): # Check if any string in series is non-empty
                 new_df[col_name] = combined_series
    
    return new_df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to known formats after merging."""
    df = df.copy() # Avoid modifying inplace
    name_mappings = {
        # Maps common variations to standard names (uppercase)
        'TEST': 'TESTS', 'TEST NAME': 'TESTS', 'EXAMINATION': 'TESTS',
        'PARAMETER': 'TESTS', 'ANALYTE': 'TESTS',
        'VALUE': 'RESULT', 'RESULTS': 'RESULT', 'MEASUREMENT': 'RESULT',
        'REFERENCE': 'REFERENCE INTERVAL', 'REF INTERVAL': 'REFERENCE INTERVAL',
        'REF RANGE': 'REFERENCE INTERVAL', 'REFERENCE RANGE': 'REFERENCE INTERVAL',
        'NORMAL RANGE': 'REFERENCE INTERVAL', 'NORMAL VALUES': 'REFERENCE INTERVAL',
        'UNIT': 'UNITS',
        'FLAGS': 'FLAG', 'ABNORMAL FLAG': 'FLAG',
    }

    # Clean -> Uppercase -> Map
    df.columns = [clean_column_name(str(col)).upper() for col in df.columns]
    df.columns = [name_mappings.get(col, col) for col in df.columns] # Keep original if no mapping
    return df


def clean_table_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and standardize table data extracted by docling. Includes detailed debug logging
    and logic to promote the first row to header if columns are default integers.

    Args:
        df: The raw DataFrame extracted from a table.

    Returns:
        A tuple: (original DataFrame with uniquely named columns, cleaned DataFrame).
                 Returns empty DataFrames if input is invalid or empty.
    """

    if df.empty:
        _log.debug("clean_table_data received an empty DataFrame.")
        original_cols = rename_duplicate_columns(df.columns)
        return pd.DataFrame(columns=original_cols), pd.DataFrame()

    # --- Log Initial State ---
    _log.debug(f"--- clean_table_data: Input DataFrame ---")
    _log.debug(f"Input columns: {df.columns.tolist()}")
    _log.debug(f"Input head:\n{df.head().to_string()}")
    # ---

    # <<< START NEW HEADER PROMOTION LOGIC >>>
    # Check if columns are default integers (0, 1, 2...)
    is_default_header = all(isinstance(col, int) for col in df.columns)

    if is_default_header and not df.empty:
        _log.debug("Default integer columns detected. Promoting first row to header.")
        # Store the first row as potential headers
        new_headers = df.iloc[0].astype(str).tolist() # Convert first row to list of strings
        # Remove the first row from the DataFrame
        df = df.iloc[1:].reset_index(drop=True)
        # Assign the new headers
        df.columns = new_headers
        _log.debug(f"--- clean_table_data: After Header Promotion ---")
        _log.debug(f"Promoted columns: {df.columns.tolist()}")
        _log.debug(f"DataFrame head after promotion:\n{df.head().to_string()}")
    elif not df.empty:
         # Ensure headers are strings for consistency if not promoted
         df.columns = df.columns.astype(str)
         _log.debug("Using existing headers (converted to string).")
    # <<< END NEW HEADER PROMOTION LOGIC >>>


    # Keep original df unmodified by working on copies *after* potential header promotion
    df_copy = df.copy()

    # Create a display version with duplicates renamed early on
    original_display_df = df_copy.copy()
    original_display_df.columns = rename_duplicate_columns(df_copy.columns) # Use potentially promoted headers
    # --- Log Original Display State ---
    _log.debug(f"--- clean_table_data: Original Display DataFrame (renamed duplicates) ---")
    _log.debug(f"Original Display columns: {original_display_df.columns.tolist()}")
    _log.debug(f"Original Display head:\n{original_display_df.head().to_string()}")
    # ---

    try:
        # 1. Merge columns based on cleaned original names (using potentially promoted headers)
        _log.debug("Attempting column merging...")
        df_merged = merge_duplicate_columns(df_copy)
        # --- Log Merged State ---
        _log.debug(f"--- clean_table_data: Merged DataFrame ---")
        _log.debug(f"Merged columns: {df_merged.columns.tolist()}")
        _log.debug(f"Merged head:\n{df_merged.head().to_string()}")
        # ---

        # 2. Standardize column names of the merged DataFrame
        _log.debug("Attempting column name standardization...")
        cleaned_df = standardize_column_names(df_merged)
        # --- Log Standardized State ---
        _log.debug(f"--- clean_table_data: Standardized DataFrame ---")
        _log.debug(f"Standardized columns: {cleaned_df.columns.tolist()}")
        _log.debug(f"Standardized head:\n{cleaned_df.head().to_string()}")
        # ---
        
        # 3. Define preferred column order and apply it
        _log.debug("Applying final column order...")
        standard_columns = ['TESTS', 'RESULT', 'FLAG', 'UNITS', 'REFERENCE INTERVAL', 'LAB']
        existing_ordered_columns = [col for col in standard_columns if col in cleaned_df.columns]
        other_columns = sorted([col for col in cleaned_df.columns if col not in standard_columns])
        final_column_order = existing_ordered_columns + other_columns
        # Ensure all original columns are included if reindexing
        cleaned_df = cleaned_df.reindex(columns=final_column_order, fill_value=None)
        # --- Log Final State ---
        _log.debug(f"--- clean_table_data: Final Cleaned DataFrame ---")
        _log.debug(f"Final columns: {cleaned_df.columns.tolist()}")
        _log.debug(f"Final head:\n{cleaned_df.head().to_string()}")
        # ---
    
    except Exception as e:
        _log.exception(f"Error during table cleaning: {e}")
        # Fallback: return the display version as both if cleaning fails critically
        return original_display_df, original_display_df.copy()

    return original_display_df, cleaned_df

# --- Biomarker Extraction Functions ---
# (Keep extract_biomarkers_from_text, clean_table_text, get_unit_type as they are general utilities)
def extract_biomarkers_from_text(text_content: str) -> Optional[pd.DataFrame]:
    """
    Extract biomarker information from cleaned text content using regex.

    Args:
        text_content: The cleaned text potentially containing biomarker data.

    Returns:
        A pandas DataFrame with extracted biomarkers, or None if no text or biomarkers found.
    """
    if not text_content or not isinstance(text_content, str):
        _log.info("extract_biomarkers_from_text received empty or invalid input.")
        return None

    # It's often better to clean the text *before* passing it here.
    # text_content = clean_table_text(text_content) # Assuming pre-cleaned

    biomarkers: List[Dict[str, Any]] = []
    lines = text_content.strip().split('\n')

    # --- Header Detection ---
    start_idx = 0
    if lines:
        header_keywords = {'TEST', 'RESULT', 'UNIT', 'FLAG', 'REFERENCE', 'VALUE', 'RANGE', 'ANALYTE'}
        # Simple check: words in first line AND structure (pipe or multiple spaces)
        first_line_words = set(re.findall(r'\b\w+\b', lines[0].upper()))
        if header_keywords.intersection(first_line_words) and ('|' in lines[0] or re.search(r'\s{2,}', lines[0])):
             start_idx = 1
             _log.debug(f"Skipping potential header line: '{lines[0]}'")

    # --- Regex Patterns (Compile once) ---
    numeric_result_pattern = re.compile(r'([-+]?\s?\d{1,3}(?:,?\d{3})*(?:\.\d+)?|\.\d+)\b(?:[eE][-+]?\d+)?') # Added optional scientific notation
    range_result_pattern = re.compile(r'([<>≤≥]=?\s?\d+\.?\d*|\d+\.?\d*\s?-\s?\d+\.?\d*)\b') # Optional equals for inequalities
    # Combine qualitative results into one pattern
    qualitative_result_pattern = re.compile(r'\b(Positive|Negative|Detected|Not Detected|Normal|Abnormal|Reactive|Non-Reactive|Present|Absent|High|Low)\b', re.IGNORECASE) # Added more common terms

    # Flag patterns (often single chars or symbols near results)
    flag_pattern = re.compile(r'\b([HLAN]|[<>*]+)\b', re.IGNORECASE) # H,L,A,N or symbols like *, **, <, >

    # Reference interval pattern (more flexible)
    ref_interval_pattern = re.compile(r'(\(?\s?<?>?\s?\d+\.?\d*\s?-\s?<?>?\s?\d+\.?\d*\s?\)?|<[=>]?\s?\d+\.?\d*|>[=>]?\s?\d+\.?\d*|\b(?:up\s+to|below|above)\s+\d+\.?\d*\b|\b[A-Za-z\s]+(?:range|limit)\b)', re.IGNORECASE)


    # --- Line Parsing Logic ---
    for line_num, line in enumerate(lines[start_idx:], start=start_idx):
        line = line.strip()
        if not line: continue

        # Split line into potential columns
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
        else:
            # Use broader whitespace split, but be aware it might merge test names
            parts = [p.strip() for p in re.split(r'\s{2,}', line)] # Split on 2+ spaces

        parts = [p for p in parts if p] # Remove empty strings
        if not parts: continue

        # --- Data Extraction within Line ---
        test_name = parts[0] # Assume first part is the test name
        result_str: Optional[str] = None
        unit: Optional[str] = None
        flag: Optional[str] = None
        reference: Optional[str] = None

        potential_value_parts = parts[1:] # Parts after assumed test name
        extracted_indices = set() # Track used parts to avoid double assignment

        # Strategy: Identify the most likely candidates for each field
        # 1. Find Result (Numeric > Range > Qualitative)
        for i, part in enumerate(potential_value_parts):
            if i in extracted_indices: continue
            num_match = numeric_result_pattern.search(part)
            range_match = range_result_pattern.search(part)
            qual_match = qualitative_result_pattern.search(part)
            # Prioritize numeric, then range, then qualitative if multiple could match
            if num_match:
                result_str = num_match.group(1).replace(',', '')
                extracted_indices.add(i)
                break # Found a numeric result, assume it's the main one
            elif range_match and result_str is None: # Only take range if no numeric found yet
                result_str = range_match.group(1)
                extracted_indices.add(i)
            elif qual_match and result_str is None: # Only take qualitative if no numeric/range
                result_str = qual_match.group(1)
                extracted_indices.add(i)
        # If a result was found, break early? Or continue searching for unit/flag? Let's continue searching.

        # 2. Find Unit (often near result or in its own column)
        for i, part in enumerate(potential_value_parts):
            # Skip if part was already identified as the primary result holder
            if i in extracted_indices and len(potential_value_parts) > 1: continue
            # Allow unit check even on result part if it contains both (e.g. "10.5 mg/dL")
            unit_match = unit_pattern.search(part)
            if unit_match:
                unit = unit_match.group(1)
                extracted_indices.add(i)
                break # Found a unit, stop searching for unit

        # 3. Find Flag (can be standalone or appended/prepended to result/unit)
        for i, part in enumerate(potential_value_parts):
            if i in extracted_indices and len(potential_value_parts) > 1: continue
            # Check full part first
            flag_match = flag_pattern.fullmatch(part)
            if flag_match:
                flag = flag_match.group(1)
                extracted_indices.add(i)
                break # Found flag, stop searching
            # Check if flag is at the end of the result string if not standalone
            if result_str and not flag and i not in extracted_indices:
                 potential_flag_in_result = flag_pattern.search(part) # Search within part
                 if potential_flag_in_result:
                      # Simple check: is the flag at the very end?
                      if part.endswith(potential_flag_in_result.group(1)):
                           flag = potential_flag_in_result.group(1)
                           # Optional: remove flag from result_str if needed?
                           # result_str = result_str[:-len(flag)].strip()
                           extracted_indices.add(i) # Mark part as used
                           break


        # 4. Find Reference Interval (usually its own column or after result/unit)
        for i, part in enumerate(potential_value_parts):
            if i in extracted_indices: continue # Skip parts already used for result/unit/flag
            ref_match = ref_interval_pattern.search(part)
            # Basic check: does it look like a range/limit and isn't identical to the result?
            if ref_match and (result_str is None or ref_match.group(1).strip() != result_str.strip()):
                reference = ref_match.group(1).strip()
                extracted_indices.add(i)
                break # Found reference, stop searching

        # --- Append Biomarker Data ---
        if test_name and result_str is not None: # Require at least a test name and some result
            # Final cleanup
            test_name = test_name.strip(':*-. ').strip() # Remove common trailing chars
            unit_type = get_unit_type(unit) if unit else 'None'

            biomarkers.append({
                'Test': test_name,
                'Result': result_str, # Store the raw extracted string
                'Units': unit or '',
                'Unit Type': unit_type,
                'Flag': flag or '',
                'Reference Interval': reference or ''
            })
        elif test_name:
             _log.debug(f"Line {line_num+1}: Found test name '{test_name}' but failed to extract result from parts: {potential_value_parts}")


    # --- Post-processing and DataFrame Creation ---
    if not biomarkers:
        _log.warning("No biomarkers extracted from the provided text content.")
        return None

    df = pd.DataFrame(biomarkers)

    # Attempt numeric conversion into a separate column for calculations/filtering
    # Handle potential errors during conversion (e.g., '<5', 'Detected')
    df['Result_Numeric'] = pd.to_numeric(
        df['Result'].astype(str).str.replace(',', '', regex=False).str.strip('<>≤≥= '),
        errors='coerce' # Non-numeric results become NaN
    )

    # Clean Units column (remove parentheses, extra spaces)
    df['Units'] = df['Units'].astype(str).str.strip('() ').str.strip()

    _log.info(f"Successfully extracted {len(df)} potential biomarkers.")
    return df


def clean_table_text(text_content: str) -> str:
    """Clean table-like text content by normalizing separators and removing noise."""
    if not isinstance(text_content, str): return ""

    cleaned_lines: List[str] = []
    for line in text_content.splitlines(): # Handles \n, \r, \r\n
        line = line.strip()
        if not line: continue # Skip empty lines

        # Normalize pipe separators, handling spaces around them
        line = re.sub(r'\s*\|\s*', '|', line) # Replace ' | ' with '|'
        line = re.sub(r'\|+', '|', line)      # Replace '||' with '|'
        line = line.strip('| ')               # Remove leading/trailing pipes/spaces

        # Remove markdown table separator lines (e.g., |---|---| or :---:|---:)
        # Allows for spaces within separators like | :-- | --: |
        if re.fullmatch(r'\|?[-:|\s]+\|?', line):
            continue

        # Remove lines that look like simple page numbers or boilerplate footers
        if re.fullmatch(r'Page\s+\d+(\s+(of|/)\s+\d+)?', line, re.IGNORECASE):
             continue
        # Add more footer/header patterns if needed

        # Only add lines that contain actual content beyond separators
        if line.replace('|', '').strip():
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def get_unit_type(unit: Optional[str]) -> str:
    """Determine if a unit is Standard, SI, or Other based on predefined lists."""
    if not unit or not isinstance(unit, str):
        return 'Other' # Treat None or non-string as Other

    # Define sets for efficient lookup (lowercase, no spaces)
    # Consider loading these from a config file if they grow large
    STANDARD_UNITS = {
        'ng/l', 'mg/g', 'ug/dl', 'mg/dl', 'u/l', 'mg/l', 'ug/l', 'ng/ml', 'ug/ml',
        'g/dl', 'pg/ml', 'mm[hg]', 'mmhg', 'iu/l', 'g/cm2', 'kg', '%', 'ω', 'l',
        '1000cells/ul', 'cells/ul', '/ul', '/μl', 'fl', 'mosm/kg', '10*3/ul', '10^3/ul', 'x10^3/ul',
        'x10^12/l', '10*12/l', '10^12/l', 'uiu/l', 'ng/dl', 'ml', 'ug/g[hb]', 'ug/ghb',
        'mm/h', 'mm/hr', 'cm', 'kpa', 'uiu/ml', 'iu/ml', 'miu/ml', 'type', 'present',
        'positive', 'negative', 'detected', 'notdetected', 'normal', 'abnormal',
        'reactive', 'nonreactive', '10*6/ml', '10^6/ml', 'x10^6/ml', 'score',
        'ml/min/1.73m2', 'ml/min/1.73m²', 'g/g{creat}', 'g/gcreat', 'ratio', 'mg/ml',
        'ph', 'clarity', 'color', 'kcal/kg/h', 'nm', 'pattern', 'count', 's', 'sec',
        'min', 'hr', "arb'u/ml", 'index' # Added index, /μl, corrected powers
    }
    SI_UNITS = {
        'nmol/l', 'mg/mmol', 'umol/l', 'mmol/l', 'nmol/ml', 'pmol/l', 'fmol/l' # Added fmol/l
    }

    # Clean and standardize the input unit string
    unit_clean = unit.lower().strip()
    unit_clean = re.sub(r'\s+', '', unit_clean)  # Remove all internal whitespace
    unit_clean = unit_clean.strip('()[]{}')     # Remove surrounding brackets/parens

    # Check against predefined sets
    if unit_clean in STANDARD_UNITS:
        return 'Standard'
    elif unit_clean in SI_UNITS:
        return 'SI'
    else:
        # Optional: Log unknown units for review
        # if len(unit_clean) > 1: # Avoid logging single characters potentially misidentified as units
        #     _log.debug(f"Unit '{unit}' classified as Other (cleaned: '{unit_clean}')")
        return 'Other'

# --- Revised Date Extraction Function (Full Text Logic - v2) ---

def extract_lab_date(conv_res: Any) -> Optional[date_type]:
    """
    Extracts and parses a lab date from docling ConversionResult,
    prioritizing keywords found in the full exported text. Handles dates
    on the same line or the immediately following line.

    Searches the full text export for keywords. If a keyword is found,
    it looks for a date pattern immediately following it on the same line,
    or at the beginning of the next line.
    Prioritizes keywords:
    1. "Date reported", "Reported", "Date Entered", "Time Entered"
    2. "Date collected", "Collected", "Date and Time Collected"
    Parses the found string into a datetime.date object.

    Args:
        conv_res: The docling ConversionResult object.

    Returns:
        The extracted and parsed date object based on the highest priority keyword,
        or None if no matching date is found or parsing fails.
    """
    if not conv_res or not hasattr(conv_res, 'document'):
        _log.warning("extract_lab_date: Invalid or missing ConversionResult/document.")
        return None

    # --- Get Full Text ---
    full_text: str = ""
    try:
        full_text = conv_res.document.export_to_text()
        if not full_text:
            _log.warning("extract_lab_date: Document text export returned empty.")
            return None
        # _log.debug(f"--- extract_lab_date: Full Text --- \n{full_text[:1000]}...")
    except Exception as text_exc:
        _log.error(f"extract_lab_date: Failed to export document to text: {text_exc}")
        return None

    # --- Keyword and Date Patterns ---
    keywords_priority1_patterns = [
        r"date\s+reported", r"reported",
        r"date\s+entered", r"time\s+entered"
    ]
    keywords_priority2_patterns = [
        r"date\s+collected", r"collected",
        r"date\s+and\s+time\s+collected"
    ]
    keyword_search_pattern = re.compile(
        r"(?i)\b(" + "|".join(keywords_priority1_patterns + keywords_priority2_patterns) + r")\b\s*:?"
    )
    # Date pattern includes the fix for missing space between date and time
    date_pattern = re.compile(r"""
        (                           # Start capturing group 1: Full date/datetime string found
            (?:
                \b\d{1,2}[/\-. ]\d{1,2}[/\-.]\d{2,4}\b
                |
                \b\d{1,2}\s+[A-Za-z]{3,}\s+\d{4}\b
                |
                \b[A-Za-z]{3,}\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b
                |
                \b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b
            )
            (?:                     # Optional Time Part (with space)
                \s+                 # Require space separator
                \d{1,2}:\d{2}       # HH:MM
                (?::\d{2})?         # Optional :SS
                (?:\s*(?:AM|PM))?   # Optional AM/PM
            )?
            | # OR: Handle case like 08/26/1600:00 (no space)
            (?:(?:\b\d{1,2}[/\-. ]\d{1,2}[/\-.]\d{2,4})(\d{2}:\d{2}(?::\d{2})?)) # Date immediately followed by HH:MM(:SS)

        )                           # End capturing group 1
        """, re.VERBOSE | re.IGNORECASE)

    # --- Search Full Text ---
    found_dates: Dict[int, str] = {}
    MAX_CHARS_TO_CHECK_ON_SAME_LINE = 30  # How many chars after keyword on same line
    MAX_CHARS_TO_CHECK_ON_NEXT_LINE = 50  # How many chars at start of next line

    for match in keyword_search_pattern.finditer(full_text):
        keyword_found = match.group(1).lower().strip()
        keyword_end_pos = match.end()
        priority = 1 if any(re.fullmatch(p, keyword_found, re.IGNORECASE) for p in keywords_priority1_patterns) else 2

        if 1 in found_dates and priority == 2: continue

        _log.debug(f"Keyword '{keyword_found}' (P{priority}) found ending at char index {keyword_end_pos}")

        date_str_found: Optional[str] = None

        # 1. Check on the same line immediately after the keyword
        same_line_search_area = full_text[keyword_end_pos : keyword_end_pos + MAX_CHARS_TO_CHECK_ON_SAME_LINE]
        # Ensure we don't cross a line break when checking same line
        line_break_pos_in_area = same_line_search_area.find(chr(10))
        if line_break_pos_in_area != -1:
            same_line_search_area = same_line_search_area[:line_break_pos_in_area]

        _log.debug(f"  Searching same line in: '{same_line_search_area.strip()}'")
        same_line_date_match = date_pattern.search(same_line_search_area)
        if same_line_date_match: #and same_line_date_match.start() < 20: # Check if starts very close
            date_str_found = same_line_date_match.group(1).strip()
            _log.debug(f"  Found date match on SAME line: '{date_str_found}'")

        # 2. If not found on same line, check the beginning of the next line
        if not date_str_found:
            next_line_start_pos = full_text.find(chr(10), keyword_end_pos)
            if next_line_start_pos != -1: # Found a newline character after the keyword
                # Skip potential empty lines
                next_line_start_pos += 1
                while next_line_start_pos < len(full_text) and full_text[next_line_start_pos] in ('\r', '\n', ' '):
                     next_line_start_pos += 1

                if next_line_start_pos < len(full_text):
                    next_line_search_area = full_text[next_line_start_pos : next_line_start_pos + MAX_CHARS_TO_CHECK_ON_NEXT_LINE]
                    _log.debug(f"  Searching next line in: '{next_line_search_area.strip()}'")
                    next_line_date_match = date_pattern.search(next_line_search_area)
                    # Check if the date match starts right at the beginning of the next line text
                    if next_line_date_match and next_line_date_match.start() < 5: # Allow few spaces
                         date_str_found = next_line_date_match.group(1).strip()
                         _log.debug(f"  Found date match on NEXT line: '{date_str_found}'")


        # 3. Process found date string (if any)
        if date_str_found:
            # Handle the specific "DateHH:MM" case by adding space if needed by dateparser
            # Use a more specific regex for this check to avoid altering already correct strings
            no_space_match = re.fullmatch(r"(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})(\d{2}:\d{2}(?::\d{2})?)", date_str_found)
            if no_space_match:
                date_part = no_space_match.group(1)
                time_part = no_space_match.group(2)
                date_str_found = f"{date_part} {time_part}"
                _log.debug(f"  Added space for combined date-time: '{date_str_found}'")

            # Store if it's higher priority or first found for this priority
            if priority == 1 or priority not in found_dates:
                found_dates[priority] = date_str_found


    # --- Select and Parse Best Date ---
    final_date_str: Optional[str] = None
    if 1 in found_dates:
        final_date_str = found_dates[1]
        _log.info(f"Selected priority 1 date string from full text: '{final_date_str}'")
    elif 2 in found_dates:
        final_date_str = found_dates[2]
        _log.info(f"Selected priority 2 date string from full text: '{final_date_str}'")
    else:
        _log.warning("Could not find any specified date keywords followed by a recognizable date string in the full text.")
        return None

    # Parse the selected date string
    parsed_date: Optional[date_type] = None
    if final_date_str:
        try:
            parsed_datetime = dateparser.parse(final_date_str, settings={'PREFER_DATES_FROM': 'past', 'DATE_ORDER': 'MDY'})
            if parsed_datetime:
                parsed_date = parsed_datetime.date()
                _log.info(f"Successfully parsed date string '{final_date_str}' to date object: {parsed_date}")
            else:
                _log.warning(f"dateparser could not parse the final selected date string: '{final_date_str}'")
        except Exception as e:
            _log.error(f"Error parsing final date string '{final_date_str}' with dateparser: {e}")

    return parsed_date

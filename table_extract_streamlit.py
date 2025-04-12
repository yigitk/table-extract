import streamlit as st
import logging

from utils_streamlit import (
    process_document,
    show_file_preview,
    process_tables_and_biomarkers,
    extract_biomarkers_from_text
)

_log = logging.getLogger(__name__)


def main():
    st.title("Document Table Extractor")
    st.write("Upload a document (PDF, DOCX, or Image) to extract tables and structured content")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'tiff']
    )

    if uploaded_file:
        # Show file preview
        st.subheader("Document Preview")
        show_file_preview(uploaded_file)
        
        # Add a separator between preview and results
        st.markdown("---")
        st.subheader("Extracted Content")
        
        with st.spinner('Processing document...'):
            rapid_conv_res, easy_conv_res, doc_filename = process_document(uploaded_file)
            
            # Create tabs for different OCR results
            ocr_tab1, ocr_tab2 = st.tabs(["RapidOCR Results", "EasyOCR Results"])
            
            with ocr_tab1:
                st.subheader("RapidOCR Results")
                text_content = rapid_conv_res.document.export_to_text()
                biomarker_df = extract_biomarkers_from_text(text_content)
                process_tables_and_biomarkers(rapid_conv_res, biomarker_df, doc_filename, "rapid", text_content)
            
            with ocr_tab2:
                st.subheader("EasyOCR Results")
                text_content = easy_conv_res.document.export_to_text()
                biomarker_df = extract_biomarkers_from_text(text_content)
                process_tables_and_biomarkers(easy_conv_res, biomarker_df, doc_filename, "easy", text_content)

if __name__ == "__main__":
    main()
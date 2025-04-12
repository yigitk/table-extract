import streamlit as st
import logging

from utils import (
    process_document_for_streamlit,
    show_file_preview,
    process_tables_and_biomarkers,
    extract_biomarkers_from_text
)

_log = logging.getLogger(__name__)


def main():
    st.set_page_config(layout="wide")
    st.title("📄 Document Table & Biomarker Extractor")
    st.write("Upload a document (PDF, DOCX, or Image) to extract tables and structured biomarker content.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'tiff'],
        help="Supports PDF, Word (DOCX), PNG, JPG, JPEG, and TIFF files."
    )

    if uploaded_file:
        col1, col2 = st.columns(2)

        with col1:
            # Show file preview
            st.subheader("📄 Document Preview")
            try:
                show_file_preview(uploaded_file)
            except Exception as e:
                st.error(f"Could not generate preview: {e}")

        with col2:
            st.subheader("⚙️ Extracted Content")
            with st.spinner('Processing document... Please wait.'):
                try:
                    # Call the renamed function for Streamlit
                    rapid_conv_res, doc_filename, rapid_timings = process_document_for_streamlit(uploaded_file)

                    if rapid_conv_res:
                        # Display conversion time
                        if rapid_timings and "pipeline_total" in rapid_timings:
                            try:
                                total_time_seconds = rapid_timings["pipeline_total"].times[0]
                                st.metric(label="Processing Time", value=f"{total_time_seconds:.2f} seconds")
                            except (IndexError, AttributeError, KeyError):
                                st.info("Could not retrieve processing time.")

                        st.success("Document processed successfully!")

                        # Use the Streamlit-specific display function
                        text_content = rapid_conv_res.document.export_to_text()
                        biomarker_df = extract_biomarkers_from_text(text_content)
                        process_tables_and_biomarkers(rapid_conv_res, biomarker_df, doc_filename, "rapid", text_content)
                    else:
                         st.error("Document processing failed. Check logs for details.")

                except Exception as e:
                    _log.exception(f"Error in Streamlit main processing block for {uploaded_file.name}")
                    st.error(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    # Setup basic logging for Streamlit app if not configured elsewhere
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()
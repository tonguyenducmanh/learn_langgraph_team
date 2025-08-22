import streamlit as st
import pandas as pd
import io
from langchain_google_genai import ChatGoogleGenerativeAI

def render_batch_processor(llm: ChatGoogleGenerativeAI):
    """
    Renders the Streamlit UI for batch processing prompts from an Excel file.
    """
    with st.expander("Xử lý hàng loạt từ file Excel"):
        uploaded_file = st.file_uploader("Tải lên file Excel (.xlsx)", type=["xlsx"])
        has_header = st.checkbox("File của tôi có dòng tiêu đề", value=True)

        if uploaded_file is not None:
            try:
                if has_header:
                    df = pd.read_excel(uploaded_file, header=0)
                else:
                    df = pd.read_excel(uploaded_file, header=None)
                    df.columns = ["Prompt"] # Assign a default column name

                if df.empty or df.columns.empty:
                    st.error("File Excel rỗng hoặc không có cột nào.")
                else:
                    prompt_column = df.columns[0]
                    prompts = df[prompt_column].dropna().tolist()
                    
                    if not prompts:
                        st.warning("Không tìm thấy prompt nào trong cột đầu tiên.")
                    else:
                        st.info(f"Tìm thấy {len(prompts)} yêu cầu. Bắt đầu xử lý...")
                        stories = []
                        progress_bar = st.progress(0)
                        
                        for i, prompt in enumerate(prompts):
                            story_response = llm.invoke(f"Bạn là một người kể chuyện. Hãy viết một câu chuyện dựa trên yêu cầu sau: '{prompt}'")
                            stories.append(story_response.content)
                            progress_bar.progress((i + 1) / len(prompts))
                            
                        df["Nội dung truyện"] = stories
                        st.success("Xử lý hoàn tất!")
                        
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Kết quả')
                        
                        st.download_button(
                            label="Tải file kết quả",
                            data=output.getvalue(),
                            file_name="ket_qua_tao_truyen.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            except Exception as e:
                st.error(f"Đã có lỗi xảy ra khi xử lý file: {e}")

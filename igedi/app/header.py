import base64
import json
import streamlit as st

from igedi.app.config import FEEDBACK_URL, IGEDI_CITATION, LOGO_PATH
from igedi.app.state import load_example

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def citation_button():
    st.iframe(f"""
        <button id="cite" title="Copy the iGEDI citation to the clipboard"
                style="font-family: 'Source Sans Pro', sans-serif; font-size: 14px; color: #808495;
                       background: transparent; border: 1px solid rgba(128, 128, 128, 0.4);
                       border-radius: 6px; padding: 4px 10px; cursor: pointer;">Copy citation</button>
        <script>
            const citation = {json.dumps(IGEDI_CITATION)};
            const button = document.getElementById("cite");
            button.onclick = async () => {{
                try {{
                    await navigator.clipboard.writeText(citation);
                }} catch (error) {{
                    const area = document.createElement("textarea");
                    area.value = citation;
                    document.body.appendChild(area);
                    area.select();
                    document.execCommand("copy");
                    area.remove();
                }}
                button.textContent = "Copied!";
                setTimeout(() => button.textContent = "Copy citation", 2000);
            }};
        </script>
        """, height=40)

def play_header():
    logo_base64 = get_base64_image(LOGO_PATH)

    head1, head2 = st.columns([1,8])
    head1.markdown(
        f"""
        <style>
        .header-logo {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }}
        .header-logo img {{
            max-width: 100%; /* Adjust the size as needed */
            overflow: hidden;
            object-fit: contain;
            margin-top: 12px;
            padding: 6px;
            background: #ffffff;
            border-radius: 6px;
       }}
        </style>
        <div class="header-logo">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo">
        </div>
        """,
        unsafe_allow_html=True
    )
    with head2:
        st.markdown("""
        # interactive GEDI
        """)
    st.markdown("""
    ## **G**enerating **E**vent **D**ata with **I**ntentional Features for Benchmarking Process Mining
    """)
    st.markdown(
        "Real event logs rarely cover the characteristics you want to evaluate on. "
        "iGEDI generates event logs that hit **target feature values you choose** "
        "(e.g. variant entropy, trace length statistics), either a single log or a whole "
        "grid that spans the feature space evenly. Pick features and values below, press "
        "**Run Generation**, and download the `.xes` logs."
    )
    paper_col, cite_col = st.columns([6, 1], vertical_alignment="center")
    with paper_col:
        st.markdown(
            "iGEDI is based on [the ICPM'24 Demo Paper](https://ceur-ws.org/Vol-3783/paper_349.pdf). "
            "The code is available on [GitHub](https://github.com/andreamalhera/gedi). " \
            "For larger runs, prepare your `config-file.json` here and run it locally with `pip install gedi`. "
            f"Questions or ideas? [Share feedback]({FEEDBACK_URL}).",
        )
    with cite_col:
        citation_button()
    st.button("Load example: 3×3 grid over variant entropy × most common variant ratio",
              on_click=load_example)
    return

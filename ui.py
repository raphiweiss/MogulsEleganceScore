import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 0.15rem !important;
                padding-left: 1.8rem !important;
                padding-right: 1.8rem !important;
                max-width: 100%;
            }

            section[data-testid="stSidebar"] {
                background-color: #f8fafc;
                border-right: 1px solid #e5e7eb;
                min-width: 250px !important;
                max-width: 250px !important;
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
                padding-left: 0.85rem !important;
                padding-right: 0.85rem !important;
            }

            button[kind="header"],
            [data-testid="stSidebarCollapseButton"] {
                display: none !important;
                visibility: hidden !important;
            }

            h1, h2, h3, h4 {
                line-height: 1.15 !important;
                margin-top: 0 !important;
                margin-bottom: 0.2rem !important;
                white-space: normal !important;
                overflow-wrap: break-word !important;
                word-break: break-word !important;
            }

            .stPlotlyChart {
                margin-bottom: 0 !important;
            }

            /* Sidebar kompakter */
            section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
                gap: 0.06rem !important;
            }

            section[data-testid="stSidebar"] .stMarkdown p {
                margin-bottom: 0.1rem !important;
            }

            section[data-testid="stSidebar"] label {
                font-size: 0.70rem !important;
                color: #6b7280 !important;
            }

            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] .stSelectbox,
            section[data-testid="stSidebar"] .stToggle,
            section[data-testid="stSidebar"] .stMultiSelect,
            section[data-testid="stSidebar"] .stSegmentedControl {
                font-size: 0.74rem !important;
            }

            section[data-testid="stSidebar"] .stSelectbox label,
            section[data-testid="stSidebar"] .stToggle label,
            section[data-testid="stSidebar"] .stMultiSelect label,
            section[data-testid="stSidebar"] .stSegmentedControl label {
                margin-bottom: 0.08rem !important;
                color: #4b5563 !important;
            }

            section[data-testid="stSidebar"] .stSelectbox > div,
            section[data-testid="stSidebar"] .stMultiSelect > div,
            section[data-testid="stSidebar"] .stSegmentedControl > div {
                margin-bottom: 0.08rem !important;
            }

            section[data-testid="stSidebar"] [data-baseweb="select"] > div,
            section[data-testid="stSidebar"] [data-baseweb="base-input"] > div {
                min-height: 32px !important;
                border-radius: 10px !important;
                border-color: #e5e7eb !important;
                font-size: 0.86rem !important;
            }

            section[data-testid="stSidebar"] [data-baseweb="select"] span,
            section[data-testid="stSidebar"] input,
            section[data-testid="stSidebar"] .stMultiSelect span {
                font-size: 0.88rem !important;
            }

            section[data-testid="stSidebar"] [data-baseweb="tag"] {
                font-size: 0.72rem !important;
                padding: 1px 5px !important;
            }

            /* Methodik-Button in der Sidebar */
            section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
                height: 40px !important;
                min-height: 40px !important;
                border: 1px solid #d1d5db !important;
                border-radius: 10px !important;
                background: #f3f4f6 !important;
                color: #374151 !important;
                font-size: 12px !important;
                font-weight: 500 !important;
                padding: 0 14px !important;
                margin: 0 !important;
                box-shadow: none !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stButton"] > button > div {
                width: 100% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                text-align: center !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stButton"] > button p,
            section[data-testid="stSidebar"] div[data-testid="stButton"] > button span {
                width: 100% !important;
                margin: 0 !important;
                color: #374151 !important;
                font-size: 12px !important;
                font-weight: 500 !important;
                opacity: 1 !important;
                text-align: center !important;
            }

            section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover,
            section[data-testid="stSidebar"] div[data-testid="stButton"] > button:focus,
            section[data-testid="stSidebar"] div[data-testid="stButton"] > button:active {
                background: #e5e7eb !important;
                color: #111827 !important;
                border: 1px solid #cbd5e1 !important;
                box-shadow: none !important;
            }

            .sidebar-title-card {
                background: linear-gradient(90deg, #1f2937, #374151);
                color: white;
                padding: 10px 14px;
                border-radius: 10px;
                margin-top: -15px !important;
                margin-bottom: 4px !important;
            }

            .sidebar-title-main {
                font-size: 18px;
                font-weight: 600;
                line-height: 1.2;
            }

            .sidebar-title-sub {
                font-size: 14px;
                opacity: 0.8;
                margin-top: 3px;
            }

            .sidebar-section {
                font-size: 0.92rem;
                font-weight: 600;
                color: #6b7280;
                margin-top: 0.15rem;
                margin-bottom: 0.15rem;
            }

            .mes-card {
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 0.4rem 0.6rem;
                background: white;
                min-height: 220px;
            }

            .mes-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.06rem 0;
                font-size: 0.82rem;
            }

            .mes-label {
                color: #374151;
            }

            .mes-value {
                color: #111827;
                font-variant-numeric: tabular-nums;
                font-weight: 500;
                text-align: right;
                min-width: 58px;
            }

            .mes-divider {
                border-top: 1px solid #e5e7eb;
                margin: 0.2rem 0 0.1rem 0;
            }

            .mes-total {
                font-weight: 700;
            }

            .mes-official {
                color: #6b7280;
                font-size: 0.8rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_header() -> None:
    st.markdown(
        """
        <div class="sidebar-title-card">
            <div class="sidebar-title-main">Moguls Elegance Score</div>
            <div class="sidebar-title-sub">YOLOv8n · 25 fps · Full HD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_methodik_button() -> bool:
    return st.button("Methodik", use_container_width=True)

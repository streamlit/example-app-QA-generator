import streamlit as st
import pandas as pd
from annotated_text import annotated_text

# To interate through results
from collections import Counter
from pipelines import pipeline
import nltk

nltk.download("popular")

# For Download button (Johannes)
from functionforDownloadButtons import download_button
from requests_html import HTMLSession

# region Layout size

session = HTMLSession()

st.set_page_config(page_title="Q&A Generator", page_icon="🎈")

def _max_width_():
    max_width_str = f"max-width: 1700px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

# endregion Layout size 

# region Top area 

# c30, c32 = st.columns([1.9, 1])

# with c30:
#     st.image("WhatTheFaq.png", width=480)
#     st.header("")
# 
# with c32:
#     st.header("")
#     st.text("")
#     st.header("")
#     st.markdown(
#         "###### Made in [![this is an image link](https://i.imgur.com/iIOA6kU.png)](https://www.streamlit.io/)&nbsp, with :heart: by [@DataChaz](https://www.charlywargnier.com/) &nbsp | &nbsp [![this is an image link](https://i.imgur.com/thJhzOO.png)](https://www.buymeacoffee.com/cwar05)"
#     )
#     st.text("")

st.image(
    "https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/120/apple/285/person-raising-hand_1f64b.png",
    width=100,
)


st.title("Q&A Generator")

st.write("This Q&A Generator leverages the power of [Google T5 Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) to generate quality question/answer pairs from content fetched from URLs")

st.header("")

c3, c4, c5 = st.columns([1, 6, 1])

with c4:

    with st.form("Form"):

        URLBox = st.text_input("👇 Paste a URL below to get started!", autocomplete = "text", placeholder="e.g. https://streamlit.io/", help="Do not forget the https://")
        cap = 1000

        submitted = st.form_submit_button("Get your Q&A pairs")

    c = st.container()

    if not submitted and not URLBox:
        st.stop()

    if submitted and not URLBox:
        st.warning("☝️ Please add a URL")
        st.stop()

selector = "p"

try:
    with session.get(URLBox) as r:
        paragraph = r.html.find(selector, first=False)
        text = " ".join([p.text for p in paragraph])

except:
    c.error(
        "🚫 The URL seems invalid. Please ensure you've added 'https://' or 'http://' at the start of the URL!"
    )
    st.stop()

text2 = (text[:cap] + "..") if len(text) > cap else text
lenText = len(text2)

if lenText > cap:
    c.warning(
        "We will build the Q&A pairs based on the first 1,000 characters"
    )
    pass
else:
    pass

with st.expander(" ↕️ Toggle to check extracted text ", expanded=False):
    st.header("")
    a = "The full text extraction is " + str(len(text)) + " characters long"
    st.header("")
    st.write(text2)
    st.header("")
    annotated_text(
        (a, "", "#8ef"),
    )
    st.header("")

try:
    nlp = pipeline("multitask-qa-qg")
    faqs = nlp(text2)

    st.markdown("#### **Select your favourite Q&A pairs **")
    st.header("")

    from collections import Counter

    k = [x["answer"] for x in faqs]

    new_faqs = []

    for i in Counter(k):
        all = [x for x in faqs if x["answer"] == i]
        new_faqs.append(max(all, key=lambda x: x["answer"]))

    c19, c20 = st.columns([3, 1.8])

    a_list = []

    with c19:
        filtered_Qs = [
            item for item in new_faqs if st.checkbox(item["question"], key=100)
        ]

    with c20:
        filtered_As = [
            itemw for itemw in new_faqs if st.checkbox(itemw["answer"], key=1000)
        ]

    df = pd.DataFrame(filtered_Qs)
    df2 = pd.DataFrame(filtered_As)
    frames = [df, df2]
    result = pd.concat(frames)
    result = result.drop_duplicates(subset=["question", "answer"])
    result.index += 1

    st.header("")

    st.markdown("#### ** Download your selected Q&A pairs! **")
    st.header("")

    if result.empty:
        b = "To download your Q&A's you need to start selecting them! ☝️"
        annotated_text(
            (b, "", "#faa"),
        )

    else:
        result = result[["question", "answer"]]
        CSVButton2 = download_button(
            result, "Downloaded_Q&As.csv", "📥 Download your Q&As"
        )
        st.table(result)

except Exception as e:
    st.warning(
        f"""
    🔮 **Snap!** Seems like there's an issue with that URL, please try another one. If the issue persists, [reach me out on Gitter!](https://gitter.im/DataChaz/what-the-FAQ)
    """
    )
    st.stop()
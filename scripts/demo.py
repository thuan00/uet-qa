import os
import json
import logging
import streamlit as st
from annotated_text import annotation
from markdown import markdown

from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import ElasticsearchRetriever
from uetqa.retriever import ESSentenceTransformersRetriever, HybridRetriever
from uetqa.base import BaseReader, query

# env
INDEX_NAME = os.getenv("INDEX_NAME", "uetqa-demo")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ncthuan/vi-distilled-msmarco-MiniLM-L12-cos-v5")
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", "384"))
SIMILARITY = os.getenv("SIMILARITY", "cosine")
READER_MODEL = os.getenv("READER_MODEL", "ncthuan/xlm-l-uetqa")


st.set_page_config(
    page_title="UETQA Demo",
    page_icon="https://uet.vnu.edu.vn/wp-content/uploads/2017/02/logo2_new.png",
    layout="wide",
)


# Init
#
@st.experimental_singleton
def init_retriever_reader():
    print('init retriever')
    document_store = ElasticsearchDocumentStore(
        index=INDEX_NAME,
        analyzer='standard',
        embedding_dim=EMBEDDING_DIM,
        similarity=SIMILARITY,
    )
    sparse_retriever = ElasticsearchRetriever(
        document_store,
        top_k=10
    )
    dense_retriever = ESSentenceTransformersRetriever(
        document_store=document_store,
        top_k=10,
        embedding_model=EMBEDDING_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        progress_bar=False,
    )
    retriever = HybridRetriever(
        sparse_retriever,
        dense_retriever,
        weight_on_dense=True,
        normalization=False,
    )

    print('init reader')
    reader = BaseReader(READER_MODEL)
    reader.qa_model.to(dense_retriever.devices[0])

    return retriever, reader

retriever, reader = init_retriever_reader()


# Load sample questions
#
@st.experimental_singleton
def load_sample_questions():
    data = []
    with open('data/uetqa-train.json','r') as f:
        data.extend(json.load(f)['data'])
    with open('data/uetqa-test.json','r') as f:
        data.extend(json.load(f)['data'])
    qas_dict = {}
    for a in data:
        for p in a['paragraphs']:
            for qa in p['qas']:
                answer = 'N/A' if len(qa['answers']) == 0 else qa['answers'][0]['text']
                qas_dict[qa['question']] = answer
    return qas_dict


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

def reset_results(*args):
    # Small callback to reset the interface in case the text of the question changes
    st.session_state.answer = None
    st.session_state.results = None

# Persistent state
set_state_if_absent("question", None)
set_state_if_absent("answer", None)
set_state_if_absent("results", None)


# Title
st.write("# UETQA Demo")
st.markdown(
    """You can pose any quetion on "Quy chế đào tạo", "Quy chế công tác sinh viên", and FAQs related to UET-VNUH.
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"""
<style>
    a {{
        text-decoration: none;
    }}
    .haystack-footer {{
        text-align: center;
    }}
    .haystack-footer h4 {{
        margin: 0.1rem;
        padding:0;
    }}
    footer {{
        opacity: 0;
    }}
</style>
<div class="haystack-footer">
    <img src="https://services.uet.vnu.edu.vn/kltn/imgs/logo%20UET%202009.png" width="100" height="100">
    <h2>UETQA - <a href="https://github.com/ncthuan/uet-qa">Github</a> </h2>
    <p>Built with <a href="https://www.deepset.ai/haystack">Haystack</a> and <a href="https://streamlit.io">Streamlit</a> </p>
    <p>Get it on <a href="https://github.com/deepset-ai/haystack/">GitHub</a>  -  Read the <a href="https://haystack.deepset.ai/overview/intro">Docs</a></p>
    <p>See the <a href="https://creativecommons.org/licenses/by-sa/3.0/">License</a> (CC BY-SA 3.0).</p>
    <hr />
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Options")
top_k_reader = st.sidebar.slider(
    "Max. number of answers",
    min_value=1,
    max_value=5,
    value=3,
    step=1,
    on_change=reset_results,
)
top_k_retriever = st.sidebar.slider(
    "Max. number of documents from retriever",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    on_change=reset_results,
)
retriever_dense_weight = st.sidebar.number_input("Retriever weight on dense score", value=2.4)
reader_cls_weight = st.sidebar.number_input("Reader weight on CLS score", value=4.0)
eval_mode = st.sidebar.checkbox("Evaluation mode")




qas_dict = load_sample_questions()
questions = [''] + list(qas_dict.keys())

# Selection bar
select_question = st.selectbox("Select a sample question", questions)

# Search bar
question = st.text_input("Question", value=select_question, max_chars=200, on_change=reset_results)
# col1, col2 = st.columns(2)
# col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
# col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
st.write("___")



# Get results for query
if question is not None and len(question) > 0 and question != st.session_state.question:
    reset_results()
    st.session_state.question = question

    with st.spinner(
        "🧠 &nbsp;&nbsp; Performing neural search on documents... \n "
    ):
        try:
            st.session_state.results = query(
                retriever,
                reader,
                question,
                top_k_docs=top_k_retriever, top_k_answers=top_k_reader,
                retriever_weight=retriever_dense_weight,
                cls_weight=reader_cls_weight
            )
        except json.JSONDecodeError as je:
            st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
        except Exception as e:
            logging.exception(e)
            if "The server is busy processing requests" in str(e) or "503" in str(e):
                st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
            else:
                st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")


if st.session_state.results:

    # Show the gold answer if we use a question of the given set
    if eval_mode and st.session_state.answer:
        st.write("## Correct answer:")
        st.write(st.session_state.answer)

    # st.write("## Results:")
    for count, doc in enumerate(st.session_state.results):
        
        context = doc['content']#.replace("\n", "\n\n")

        if "answer" in doc:
            answer = doc['answer']
            answer_text = answer['answer']
            start_idx = answer['start']
            end_idx = answer['end']

            annotated_context = context[:start_idx] + str(annotation(answer_text, background="#e9ff32")) + context[end_idx:]
            # Hack due to this bug: https://github.com/streamlit/streamlit/issues/3190
            st.write(
                # markdown(context[:start_idx] + str(annotation(answer_text, background="#e9ff32")) + context[end_idx:]),
                f'<p style="white-space: pre-line;">{annotated_context}</p>',
                unsafe_allow_html=True,
            )
            # a = 
        else:
            st.write(markdown(f'<p style="white-space: pre-line;">{context}</p>'),unsafe_allow_html=True)

        # url, title = get_backlink(result)
        url = doc['meta']['href']
        title = doc['meta']['title']
        source = f"[{title}]({url})" if url else title
        st.markdown(f"**Relevance:** {round(doc['score'],2)} -  **Source:** {source}")



        if eval_mode and doc["answer"]:
            # Define columns for buttons
            is_correct_answer = None
            is_correct_document = None

            button_col1, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])
            if button_col1.button("👍", key=f"{context}{count}1", help="Correct answer"):
                is_correct_answer = True
                is_correct_document = True

            if button_col2.button("👎", key=f"{context}{count}2", help="Wrong answer and wrong passage"):
                is_correct_answer = False
                is_correct_document = False

            if button_col3.button(
                "👎👍", key=f"{context}{count}3", help="Wrong answer, but correct passage"
            ):
                is_correct_answer = False
                is_correct_document = True

            if is_correct_answer is not None and is_correct_document is not None:
                try:
                    send_feedback(
                        query=question,
                        answer_obj=result["_raw"],
                        is_correct_answer=is_correct_answer,
                        is_correct_document=is_correct_document,
                        document=result["document"],
                    )
                    st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                except Exception as e:
                    logging.exception(e)
                    # st.error("🐞 &nbsp;&nbsp; An error occurred while submitting your feedback!")
                    st.error("Feed b")

        st.write("___")


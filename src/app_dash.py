import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
# from dash import Dash, dcc, html, Input, Output

import logging
from typing import Dict, List, Optional, Union
from pprint import pprint
import pandas as pd

from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor, EmbeddingRetriever, DensePassageRetriever
from haystack.utils import convert_files_to_docs, print_answers
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import RAGenerator, Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.schema import Document

import torch
from transformers import PreTrainedTokenizer, BatchEncoding


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s") #, level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


#################### DATA PREPERATION ####################

# DATA_PATH = '../data/bitcoin_articles.csv'

# df = pd.read_csv(DATA_PATH)
# logger.info(f"DF shape: {df.shape}")

# ids = list(df["article_id"].values)
# texts = list(df["summary"].values)
# titles = list(df["title"].values)
# dates = list(df["published_date"].values)
# links = list(df["link"].values)

# all_docs = []
# for i, title, text, date, link in zip(ids, titles, texts, dates, links):
#     all_docs.append(
#         Document(
#             id=i, 
#             content=text, 
#             meta={
#                 "name": title or "", 
#                 "link": link or "", 
#                 "date": date or ""
#             }
#         )
#     )
# logger.info(len(all_docs))

# preprocessor = PreProcessor(
#     clean_empty_lines=True,
#     clean_whitespace=True,
#     clean_header_footer=False,
#     split_by="word",
#     split_length=256,
#     split_respect_sentence_boundary=True,
# )

# all_docs_process = preprocessor.process(all_docs)
# logger.info(f"n_files_input: {len(all_docs)}\nn_docs_output: {len(all_docs_process)}")

#################### DOCUMENT STORE ####################

# init only once
# doc_store_dpr = FAISSDocumentStore(sql_url = "sqlite:///faiss_document_store_dpr.db", 
#                                     faiss_index_factory_str="Flat", similarity="dot_product", return_embedding=True)
# doc_store_dpr.write_documents(all_docs_process)
# logger.info(doc_store_dpr.get_document_count())

#################### RETRIEVER ####################

# dpr_retriever = DensePassageRetriever(
#     document_store=doc_store_dpr,
#     query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
#     passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
#     use_gpu=False,
#     embed_title=True,
# )

# Add / update documents embeddings to index
# doc_store_dpr.update_embeddings(retriever=dpr_retriever)

# SAVE doc store
# doc_store_dpr.save("faiss_index_DPR.faiss")
# logger.info("Doc store DPR embeddings updated and saved!")

# LOAD doc store & retriever
document_store = FAISSDocumentStore.load("faiss_index_DPR.faiss", "faiss_index_DPR.json")

dpr_retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False,
    embed_title=True,
)
logger.info("Document store and DPR loaded!")

#################### GENERATOR ####################

class _T5Converter:
    """
    A sequence-to-sequence model input converter (https://huggingface.co/yjernite/bart_eli5) based on the T5 architecture.
    The converter takes documents and a query as input and formats them into a single sequence that a seq2seq model can use it as input for its generation step.
    This includes model-specific prefixes, separation tokens and the actual conversion into tensors.
    """
    def __call__(self, tokenizer: PreTrainedTokenizer, query: str, documents: List[Document], top_k: Optional[int] = None) -> BatchEncoding:
        conditioned_doc = "<P> " + " <P> ".join([d.content for d in documents])

        # concatenate question and support document into T5 input
        query_and_docs = "question: {} context: {}".format(query, conditioned_doc)
        max_source_length = 512
        return tokenizer([query_and_docs], truncation=True, padding=True, max_length=max_source_length, return_tensors="pt")

t5_generator = Seq2SeqGenerator(
    model_name_or_path="t5-small",
    input_converter=_T5Converter(),
    use_gpu=False,
    top_k=1,
    max_length=100,
    min_length=2,
    num_beams=2,
)
logger.info("Generator model loaded!")


#################### DASH APP & LAYOUT ####################
def textbox(text, box="other"):
    style = {
        "max-width": "55%",
        "width": "max-content",
        "padding": "10px 15px",
        "border-radius": "25px",
    }

    if box == "self":
        style["margin-left"] = "auto"
        style["margin-right"] = 0

        color = "primary"
        inverse = True

    elif box == "other":
        style["margin-left"] = 0
        style["margin-right"] = "auto"

        color = "light"
        inverse = False

    else:
        raise ValueError("Incorrect option for `box`.")

    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)


controls = dbc.InputGroup(
    style={"width": "80%", "max-width": "800px", "margin": "auto"},
    children=[
        dbc.Input(id="user-input", placeholder="Ask a question about Bitcoin...", type="text"),
        dbc.InputGroupAddon(dbc.Button("Submit", id="submit"), addon_type="append",),
    ],
)

answer = html.Div(
    style={
        "width": "80%",
        "max-width": "800px",
        "height": "70vh",
        "margin": "auto",
        "overflow-y": "auto",
    },
    id="display-conversation",
)

app = Dash(__name__)
server = app.server

app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("Bitcoin Q&A System"),
        html.Hr(),
        dcc.Store(id="store-conversation", data=""),
        controls,
        answer,
    ],
)

# app.layout = html.Div([
#     html.H5("Search a question about Bitcoin"),
#     html.Div([
#         "Input: ",
#         dcc.Input(id='qns-input', value='When was Bitcoin created?', type='text')
#     ]),
#     html.Br(),
#     html.Div(id='ans-output'),
# ])

#################### DASH CALLBACKS ####################

@app.callback(
    Output("display-conversation", "children"), 
    [Input("store-conversation", "data")]
)
def update_display(chat_history):
    return [
        textbox(x, box="self") if i % 2 == 0 else textbox(x, box="other")
        for i, x in enumerate(chat_history.split(tokenizer.eos_token)[:-1])
    ]


@app.callback(
    [Output("store-conversation", "data"), Output("user-input", "value")],
    [Input("submit", "n_clicks"), Input("user-input", "n_submit")],
    [State("user-input", "value"), State("store-conversation", "data")],
)
def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0:
        return "", ""

    if user_input is None or user_input == "":
        return chat_history, ""

    # # temporary
    # return chat_history + user_input + "<|endoftext|>" + user_input + "<|endoftext|>", ""

    # encode the new user input, add the eos_token and return a tensor in Pytorch
    bot_input_ids = tokenizer.encode(
        chat_history + user_input + tokenizer.eos_token, return_tensors="pt"
    ).to(device)

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id
    )
    chat_history = tokenizer.decode(chat_history_ids[0])

    return chat_history, ""




# @app.callback(
#     Output(component_id='ans-output', component_property='children'),
#     Input(component_id='qns-input', component_property='value')
# )
# def qna_pipeline(input_value):
#     pipe_GQA = GenerativeQAPipeline(generator=t5_generator, retriever=dpr_retriever)
    
#     result = pipe_GQA.run(query=input_value, 
#                           params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
#     return print_answers(result, details="all")


if __name__ == '__main__':
    app.run_server(debug=True)
    
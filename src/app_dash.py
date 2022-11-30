from dash import Dash, dcc, html, Input, Output

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

app = Dash(__name__)
# server = app.server

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








#################### DASH LAYOUT ####################

app.layout = html.Div([
    html.H5("Search a question about Bitcoin"),
    html.Div([
        "Input: ",
        dcc.Input(id='qns-input', value='When was Bitcoin created?', type='text')
    ]),
    html.Br(),
    html.Div(id='ans-output'),
])

#################### DASH CALLBACKS ####################

@app.callback(
    Output(component_id='ans-output', component_property='children'),
    Input(component_id='qns-input', component_property='value')
)
def qna_pipeline(input_value):
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
    
    #if t5_generator is None:
    t5_generator = Seq2SeqGenerator(
        model_name_or_path="t5-large",
        input_converter=_T5Converter(),
        use_gpu=True,
        top_k=1,
        max_length=100,
        min_length=2,
        num_beams=2,
    )
    
    pipe_GQA = GenerativeQAPipeline(generator=t5_generator, retriever=dpr_retriever)
    
    result = pipe_GQA.run(query=input_value, 
                          params={"Generator": {"top_k": 1}, "Retriever": {"top_k": 5}})
    return print_answers(result, details="all")


if __name__ == '__main__':
    app.run_server(debug=True)
    
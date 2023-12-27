from __future__ import annotations

import os
import pickle
import shutil
from pathlib import Path
from pprint import pprint
from typing import AsyncIterable

import fastapi_poe as fp
import pandas as pd
from fastapi_poe import ProtocolMessage, QueryRequest
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from tqdm import tqdm

EMBEDDING_FN = SentenceTransformerEmbeddings(model_name="msmarco-distilbert-base-tas-b")
LENGTH_FN = lambda sentence: len(EMBEDDING_FN.client.tokenizer.tokenize(sentence))
DEFAULT_BOT = "GPT-4"  # The LLM that we will using.


class PersistableMemStore(InMemoryStore):
    """filesystem saveable version of the InMemoryStore class"""

    def __init__(self, save_path):
        self.save_path = save_path

        try:
            with open(save_path, 'rb') as file:
                self.store = pickle.load(file)
        except FileNotFoundError:
            self.store = {}
        except EOFError:
            self.store = {}

    def save(self):
        with open(self.save_path, 'wb') as file:
            pickle.dump(self.store, file)


class NewsArticleQA:
    """
    Class for loading/querying a dataset with over 3000+ articles.
    Querying is done by using a vector database + communication with poe bots (ChatGPT, etc)

    Recreate the vector and keystore databases by running createdb() or begin querying by
    combining with a poe server (see ArticleQueryBot below)
    """

    def __init__(
            self,
            poe_bot,
            excel_path='NewsArticles.xlsx',
            vector_store_path='chroma_articles',
            kv_path='chroma_articles_kvstore.pkl'
    ):
        self.excel_path = str(Path(os.getcwd()) / excel_path)
        self.vs_path = str(Path(os.getcwd()) / vector_store_path)
        self.kv_path = str(Path(os.getcwd()) / kv_path)
        self.poe_bot = poe_bot

    def pdr_splitter(self):
        # default collection name for Chroma created by langchain is "langchain"
        collection_meta = {"hnsw:space": "ip"}
        vectorstore = Chroma(
            persist_directory=self.vs_path,
            embedding_function=EMBEDDING_FN,
            collection_metadata=collection_meta
        )

        # ! Max chunk_size <= model.client.max_seq_length.
        #  See: https://www.sbert.net/examples/applications/computing-embeddings/README.html#input-sequence-length
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=128,
            chunk_overlap=0,
            length_function=LENGTH_FN
        )

        # Skip splitting up the source documents as they are quite small in this article example
        # parent_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=500,
        #     chunk_overlap=0,
        #     length_function=LENGTH_FN
        # )

        docstore = PersistableMemStore(self.kv_path)
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter
        )
        return retriever, vectorstore, docstore

    async def query_db(self, request, verify_vector_results=False) -> AsyncIterable[fp.PartialResponse]:
        '''

        Find relevant articles based on the poe.com user's query, then communicate with a poe based LLM
        (GPT-3.5-Turbo, GPT-4, etc). to derive an answer.

        :param verify_vector_results - Instead of passing in all documents found via vector search, which sometimes
        yields an internal server error due to the prompt size being too large, instead pass in each document one
        at a time to the LLM and ask if it is relevant to the query. Only use relevant documents to formulate the
        final answer. Unfortunately this doesn't work very well and the LLM more often than not reports that the query
        has nothing to do with the document.
        '''

        question = request.query[-1].content
        relevant_docs = []
        docs = self.retriever.get_relevant_documents(question)
        yield fp.PartialResponse(text=f"Vector search found this many potential docs: {len(docs)}\n\n")

        if verify_vector_results:
            for doc in docs:
                if await self.doc_is_relevant_to_question(request, doc, question):
                    relevant_docs.append(doc)
                    yield fp.PartialResponse(text=f"- *{doc.metadata['title']}* was relevant\n")
                else:
                    yield fp.PartialResponse(text=f"- *{doc.metadata['title']}* was not relevant\n")
        else:
            relevant_docs = docs
            for doc in relevant_docs:
                yield fp.PartialResponse(text=f"- *{doc.metadata['title']}*\n")

        if relevant_docs:
            yield fp.PartialResponse(text="\n")
            async for msg in self.formulate_answer_with_context(request, relevant_docs):
                yield msg
        else:
            yield fp.PartialResponse(text="\nSorry, no articles matched your query...")

    def formulate_answer_with_context(self, request, relevant_docs) -> AsyncIterable[fp.PartialResponse]:
        question = request.query[-1].content
        prompt = f'''Your job is to refer to the provided content and then answer the question

To answer the question, you must follow the below rules

[RULES]
1. The provided content will have one or more documents.
2. Each document is enclosed within [DOCUMENT] and [END DOCUMENT].
3. Create the answer based on the provided documents only.
4. The answer should be as precise and concise as possible.
5. For each answer, cite the document_name and document_source that was referred to answer the question. 
6. At the end of your answer, create a list of document_name for each document you have cited.
[END RULES]

Here is a valid example for how to answer a question:

[QUESTION]
What news do you have on a recall of chicken?

[DOCUMENT]
document_name: A Million Pounds Of Chicken Recalled Over Metal Shards
document_source: http://www.huffingtonpost.com/2017/03/29/chicken-recall_n_15689952.html
document_text: Check your freezer, because you might need to throw out some contaminated chicken STAT. OK Foods Inc. 
is recalling 933,272 pounds of breaded chicken products, the U.S. Department of Agriculture's Food Safety 
and Inspection Service announced last week. The recall occurred after five people complained of discovering 
metal objects in the ready-to-eat chicken, a problem confirmed by the FSIS.
[END DOCUMENT]

A recall of chicken has been announced by the USDA's Food Safety and Inspection Service last week, due to metal 
objects found in the ready-to-eat chicken. [A Million Pounds of Chicken Recalled Over Metal Shards, 
http://www.huffingtonpost.com/2017/03/29/chicken-recall_n_15689952.html] 


[QUESTION]
{question}
'''
        for doc in relevant_docs:
            doc_string = (f"\n[DOCUMENT]\n"
                          f"document_name:{doc.metadata['title'].strip()}\n"
                          f"document_source:{doc.metadata['source']}\n"
                          f"document_text:{doc.page_content}\n"
                          f"[END DOCUMENT]\n")
            prompt += doc_string

        pprint(prompt)
        return self.msg_bot_and_stream(request, prompt)

    def msg_bot_and_stream(self, request, prompt):
        """Send a request to a Poe bot and stream the reply"""

        new_request = self.create_request(request, prompt)
        return fp.stream_request(new_request, self.poe_bot, request.access_key)

    async def msg_bot(self, request, prompt):
        """Send a request to a Poe bot and wait for the entire reply"""

        new_request = self.create_request(request, prompt)
        return ''.join([msg.text async for msg in fp.stream_request(new_request, self.poe_bot, request.access_key)])

    def create_request(self, org_request, content):
        msg = ProtocolMessage(role="user", content=content)

        request = QueryRequest(
            query=[msg],
            version=org_request.version,
            type="query",
            user_id=org_request.user_id,
            conversation_id=org_request.conversation_id,  # can be blank, but fill in anyway
            message_id="",
        )
        # pprint({'version': org_request.version,
        #         'user_id': org_request.user_id,
        #         'conversation_id': org_request.conversation_id,
        #         'message_id': org_request.message_id})

        return request

    async def doc_is_relevant_to_question(self, request, doc, msg):

        prompt = (f'Does the following document have any relationship to the question?\n'
                  f'Only answer with "yes" or "no" and nothing else.\n\n'
                  f'Document:{doc.page_content}\n\n'
                  f'Question:{msg}')
        answer = await self.msg_bot(request, prompt)
        if "yes" in answer.lower():
            return True
        else:
            pprint({'title': doc.metadata['title'], 'answer': answer})
            pprint(prompt + '\n\n')
            return False

    def load_db(self):
        self.retriever, self.vstore, self.kv_store = self.pdr_splitter()

    def create_db(self):
        docs = self.load_articles()
        if os.path.exists(self.vs_path):
            shutil.rmtree(self.vs_path)
        if os.path.exists(self.kv_path):
            os.unlink(self.kv_path)

        retriever, vstore, kv_store = self.pdr_splitter()

        chunk_size = 10
        with tqdm(total=len(docs)) as pbar:
            for i in range(0, len(docs), chunk_size):
                retriever.add_documents(docs[i:i + chunk_size], ids=None)
                pbar.update(chunk_size)
        vstore.persist()
        kv_store.save()

    def load_articles(self):
        articles = pd.read_excel(self.excel_path)
        docs = []
        for _, row in articles.iterrows():
            docs.append(
                Document(
                    page_content=row['text'],
                    metadata={
                        'date': str(row['publish_date']),
                        'source': row['article_source_link'],
                        'title': row['title'],
                    }
                )
            )
        return docs


class ArticleQueryBot(fp.PoeBot):
    def __init__(self):
        self.poe_bot = DEFAULT_BOT
        self.news_articles = NewsArticleQA(self.poe_bot)
        self.news_articles.load_db()

    def get_response(self, request: fp.QueryRequest) -> AsyncIterable[fp.PartialResponse]:
        return self.news_articles.query_db(request)

    async def get_settings(self, setting: fp.SettingsRequest) -> fp.SettingsResponse:
        return fp.SettingsResponse(server_bot_dependencies={self.poe_bot: 1})


if __name__ == '__main__':
    # NewsArticleQA().create_db()
    bot = ArticleQueryBot()
    fp.run(bot, allow_without_key=True)

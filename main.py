from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from typing import TypedDict, List
import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.documents import Document

from pdf2image import convert_from_path
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings



# LLM

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Data Loading

uploaded_data=""

class State(TypedDict):
    file_type:str
    question:str
    answer:str
    file_path:str
    documents:List[Document]
    pdf_type:str
    chunks:List[Document]
    vectorstore:object
    retrieved_docs:List[Document]
    filtered_docs:List[Document]
    retrieval_attempts: int
    sufficient: str
    critic_verdict: str
    critic_reason: str
    critic_attempts: int
    ragas_scores: dict

def filetype_node(state:State):
    
    ext = os.path.splitext(state["file_path"])[1].lower()
    if ext == ".csv":
        return {"filetype": "csv"}

    elif ext == ".pdf":
        return {"filetype": "pdf"}

    elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        return {"filetype": "image"}

    return {"filetype": "unknown"}

def route_file_type(state):

    filetype = state["filetype"]

    if filetype == "csv":
        return "csv_loader_node"

    elif filetype == "pdf":
        return "pdf_classification_node"

    elif filetype == "image":
        return "image_loader"

    return END

def csv_loader_node(state:State):

    filepath=state["file_path"]

    loader = CSVLoader(
        file_path=filepath,
        encoding="utf-8"
    )

    docs = loader.load()

    return {"documents": docs}

def pdf_classification_node(state:State):
    import pdfplumber

    filepath = state["file_path"]

    try:
        with pdfplumber.open(filepath) as pdf:
            text = ""

            for page in pdf.pages[:3]:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        if len(text.strip()) > 100:
            return {"pdf_type": "text"}

    except:
        pass

    return {"pdf_type": "scanned"}

def route_pdf_type(state):

    if state["pdf_type"] == "text":
        return "pdf_loader_node"

    return "scanned_pdf_loader_node"

def pdf_loader_node(state:State):
    loader = PyPDFLoader(state["file_path"])
    docs = loader.load()

    return {"documents": docs}

def scanned_pdf_loader_node(state:State):
    from pdf2image import convert_from_path

    images = convert_from_path(state["file_path"], 
                               poppler_path=r"C:\Users\DELL\Downloads\Release-26.02.0-0\poppler-26.02.0\Library\bin")

    text = ""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True)
    for img in images:

        temp_path = "temp.png"
        img.save(temp_path)

        result = ocr.ocr(temp_path)

        for page in result:
            for line in page:
                text += line[1][0] + "\n"

    return {
        "documents": [
            Document(page_content=text)
        ]
    }

def image_loader(state:State):

    filepath=state["file_path"]
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True)
    result = ocr.ocr(filepath)

    text = ""

    for page in result:
        for line in page:
            text += line[1][0] + "\n"

    return {"documents": [
        Document(
            page_content=text,
            metadata={"source": filepath}
        )]
    }

# Chunking

def chunking_node(state:State):
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""]
    )
    chunks = splitter.split_documents(state["documents"])

    return {"chunks": chunks}

# Storing Embedding in Vector Store


def vectorstore_node(state:State):
    vectorstore = FAISS.from_documents(
        documents=state["chunks"],
        embedding=embeddings
    )

    return {
        "vectorstore": vectorstore
    }

# Retrival

def retrieval_node(state: State):

    retrieval_attempts = state.get(
        "retrieval_attempts",
        0
    )

    critic_attempts = state.get(
        "critic_attempts",
        0
    )

    total_attempts = (
        retrieval_attempts +
        critic_attempts
    )

    k = min(
        5 + total_attempts * 5,
        20
    )

    retriever = state["vectorstore"].as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": min(
                k * 3,
                50
            )
        }
    )

    docs = retriever.invoke(
        state["question"]
    )

    return {
        "retrieved_docs": docs
    }

class GradeDocuments(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )

structured_llm = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a relevance grader.

            Your task is to determine whether a retrieved document
            contains information that can help answer the user's question.

            It does NOT need to contain the exact answer.
            It only needs to be relevant.

            Respond with:
            - yes -> relevant
            - no -> irrelevant
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Retrieved Document:
            {document}
            """
        ),
    ]
)

retrieval_grader = grade_prompt | structured_llm    

def filter_chunks_node(state:State):
    question = state["question"]
    docs = state["retrieved_docs"]

    filtered_docs = []

    for doc in docs:

        result = retrieval_grader.invoke(
            {
                "question": question,
                "document": doc.page_content
            }
        )

        if result.binary_score.lower() == "yes":
            filtered_docs.append(doc)

    return {
        "filtered_docs": filtered_docs
    }

# Checking if there are enough Document to generate answer.

class SufficiencyGrade(BaseModel):

    sufficient: str = Field(
        description="Whether the provided documents are sufficient to answer the question. Return only 'yes' or 'no'."
    )

structured_llm2 = llm.with_structured_output(
    SufficiencyGrade
)

sufficiency_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a retrieval sufficiency grader.

            Determine whether the retrieved documents contain
            enough information to answer the user's question.

            Consider:

            - Completeness
            - Coverage
            - Missing information

            Respond only:

            yes
            no
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Retrieved Documents:
            {documents}
            """
        )
    ]
)

sufficiency_chain = (
    sufficiency_prompt
    | structured_llm2
)

def check_context_sufficiency_node(state:State):

    docs_text = "\n\n".join(
        doc.page_content
        for doc in state["filtered_docs"]
    )

    result = sufficiency_chain.invoke(
        {
            "question": state["question"],
            "documents": docs_text
        }
    )

    return {
        "sufficient": result.sufficient
    }

def route_after_sufficiency(state):

    sufficient = state["sufficient"]
    attempts = state.get(
    "retrieval_attempts",0)
    
    if sufficient.lower() == "yes":
        return "generate_answer"

    if attempts >= 3:
        return "generate_answer"

    return "increase_attempts_node"

def increase_attempts_node(state: State):

     return {
        "retrieval_attempts":
            state.get(
                "retrieval_attempts",
                0
            ) + 1
    }

# Answer Generation

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the question using ONLY the provided context.

            Rules:
            - Use only information from the context.
            - Do not make up facts.
            - If the context does not contain enough information,
              clearly say so.
            - Be concise and accurate.
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)
answer_chain = answer_prompt | llm

def generation_node(state:State):

    context = "\n\n".join(
        doc.page_content
        for doc in state["filtered_docs"]
    )

    response = answer_chain.invoke(
        {
            "question": state["question"],
            "context": context
        }
    )

    return {
        "answer": response.content
    }

class CriticGrade(BaseModel):

    verdict: str = Field(
        description="""
        pass or fail
        """
    )

    reason: str = Field(
        description="""
        One of:

        sufficient_context
        insufficient_context
        hallucination
        irrelevant_answer
        """
    )
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a critic agent for a RAG system.

            Evaluate whether the answer is supported
            by the provided context and answers
            the user's question.

            Return:

            pass
            fail

            Reasons:

            sufficient_context
                -> answer is correct

            insufficient_context
                -> context does not contain enough information

            hallucination
                -> answer contains unsupported facts

            irrelevant_answer
                -> answer does not answer the question
            """
        ),
        (
            "human",
            """
            Question:
            {question}

            Context:
            {context}

            Answer:
            {answer}
            """
        )
    ]
)
critic_chain = (
    critic_prompt
    | llm.with_structured_output(
        CriticGrade
    )
)
def critic_node(state):

    context = "\n\n".join(
        doc.page_content
        for doc in state["filtered_docs"]
    )

    result = critic_chain.invoke(
        {
            "question": state["question"],
            "context": context,
            "answer": state["answer"]
        }
    )

    return {
        "critic_verdict": result.verdict,
        "critic_reason": result.reason
    }


def route_after_critic(state: State):

    verdict = state["critic_verdict"]

    if verdict == "pass":
        return "ragas_evaluation_node"

    if state.get("critic_attempts", 0) >= 3:
        return "answer_not_found"

    return "increase_critic_attempts"

def increase_critic_attempts(state: State):

    
    return {
        "critic_attempts":
            state.get(
                "critic_attempts",
                0
            ) + 1
    }

def answer_not_found_node(state: State):

    return {
        "answer": "Answer not found in the uploaded documents.",

        "final_verdict": "answer_not_found",

        "final_reason": {
            "critic_attempts": state.get("critic_attempts",0),
            "last_critic_reason": state.get(
                "critic_reason",
                "unknown"
            )
        }
    }





eval_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an evaluation agent for a RAG system.
        Given a question, context, and answer, score the answer on 4 metrics.
        Return ONLY a JSON object with these exact keys and float scores between 0.0 and 1.0:

        {{
            "faithfulness": <float>,
            "answer_relevancy": <float>,
            "context_precision": <float>,
            "context_recall": <float>
        }}

        Metric definitions:
        - faithfulness: Is the answer fully supported by the context? (1.0 = fully supported, 0.0 = hallucinated)
        - answer_relevancy: Does the answer directly address the question? (1.0 = perfectly relevant)
        - context_precision: Is the retrieved context focused and free of noise? (1.0 = very precise)
        - context_recall: Does the context contain all info needed to answer? (1.0 = complete coverage)

        Return ONLY the JSON. No explanation, no markdown, no extra text.
        """
    ),
    (
        "human",
        """
        Question: {question}

        Context: {context}

        Answer: {answer}
        """
    )
])

eval_chain = eval_prompt | llm | StrOutputParser()

def ragas_evaluation_node(state: State):
    import json

    context = "\n\n".join(
        doc.page_content
        for doc in state["filtered_docs"]
    )

    raw = eval_chain.invoke({
        "question": state["question"],
        "context": context,
        "answer": state["answer"]
    })

    try:
        clean = raw.strip().replace("```json", "").replace("```", "").strip()
        scores = json.loads(clean)
    except Exception:
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "error": "Failed to parse scores"
        }

    return {"ragas_scores": scores}
    

# Graph Construction 

builder = StateGraph(State)

# Adding Nodes

builder.add_node("filetype_node", filetype_node)
builder.add_node("route_file_type", route_file_type)

builder.add_node("csv_loader_node", csv_loader_node)

builder.add_node(
    "pdf_classification_node",
    pdf_classification_node
)

builder.add_node(
    "pdf_loader_node",
    pdf_loader_node
)

builder.add_node(
    "scanned_pdf_loader_node",
    scanned_pdf_loader_node
)

builder.add_node(
    "image_loader",
    image_loader
)

builder.add_node(
    "chunking_node",
    chunking_node
)

builder.add_node(
    "vectorstore_node",
    vectorstore_node
)

builder.add_node(
    "retrieval_node",
    retrieval_node
)

builder.add_node(
    "filter_chunks_node",
    filter_chunks_node
)

builder.add_node(
    "check_context_sufficiency_node",
    check_context_sufficiency_node
)

builder.add_node(
    "increase_attempts_node",
    increase_attempts_node
)

builder.add_node(
    "generation_node",
    generation_node
)

builder.add_node(
    "critic_node",
    critic_node
)

builder.add_node(
    "increase_critic_attempts",
    increase_critic_attempts
)

builder.add_node(
    "answer_not_found",
    answer_not_found_node
)

builder.add_node(
    "ragas_evaluation_node",
    ragas_evaluation_node
)

# Add Edges

builder.add_edge(
    START,
    "filetype_node"
)

builder.add_conditional_edges(
    "filetype_node",
    route_file_type,
    {
        "csv_loader_node":
            "csv_loader_node",

        "pdf_classification_node":
            "pdf_classification_node",

        "image_loader":
            "image_loader",

        END: END
    }
)
builder.add_conditional_edges(
    "pdf_classification_node",
    route_pdf_type,
    {
        "pdf_loader_node":
            "pdf_loader_node",

        "scanned_pdf_loader_node":
            "scanned_pdf_loader_node"
    }
)
builder.add_edge(
    "csv_loader_node",
    "chunking_node"
)

builder.add_edge(
    "pdf_loader_node",
    "chunking_node"
)

builder.add_edge(
    "scanned_pdf_loader_node",
    "chunking_node"
)

builder.add_edge(
    "image_loader",
    "chunking_node"
)
builder.add_edge(
    "chunking_node",
    "vectorstore_node"
)

builder.add_edge(
    "vectorstore_node",
    "retrieval_node"
)

builder.add_edge(
    "retrieval_node",
    "filter_chunks_node"
)

builder.add_edge(
    "filter_chunks_node",
    "check_context_sufficiency_node"
)
builder.add_conditional_edges(
    "check_context_sufficiency_node",
    route_after_sufficiency,
    {
        "generate_answer":
            "generation_node",

        "increase_attempts_node":
            "increase_attempts_node"
    }
)
builder.add_edge(
    "increase_attempts_node",
    "retrieval_node"
)
builder.add_edge(
    "generation_node",
    "critic_node"
)
builder.add_conditional_edges(
    "critic_node",
    route_after_critic,
    {
        "ragas_evaluation_node":
            "ragas_evaluation_node",

        "increase_critic_attempts":
            "increase_critic_attempts",

        "answer_not_found":
            "answer_not_found"
    }
)
builder.add_edge(
    "increase_critic_attempts",
    "retrieval_node"
)
builder.add_edge(
    "ragas_evaluation_node",
    END
)

builder.add_edge(
    "answer_not_found",
    END
)
graph = builder.compile()


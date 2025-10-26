from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from typing import Optional, List, Tuple
from langchain_chroma import Chroma
from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
import textwrap
pd.set_option("display.max_colwidth", None)

"""

Why This Matters:

This step ensures your RAG project is solving a **real information gap**, not just repeating what the model already knows.

# 1- Selected Topic

**Topic Chosen**: iOS 18, Apple’s mobile operating system released September 2024.

**Reasoning**:

- iOS 18 falls after the model’s June 2024 cutoff.

- It’s widely used, and details (new features, UI changes) are not in the base model’s training.

- A simple factual question about iOS 18 should reveal whether a knowledge base is required.

# 2- Formulate Question

**Question**:
“What are the main new features and user-interface changes introduced in iOS 18?”

**This question is:**

* Factual (no opinions).

* Specific to post-cutoff content.

- Unambiguous (“main new features and UI changes” in that single OS release).
"""

READER_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    READER_MODEL_NAME,
    quantization_config=bnb_config
    )
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

initial_question = "What are the main new features and user-interface changes introduced in iOS 18?"
base_output = READER_LLM(initial_question)
base_answer = base_output[0]["generated_text"].strip()

print(base_answer)

"""##  Evaluation of the Model’s Output
**Incomplete?**
- Yes. The model never lists any iOS 18 features—it just states that iOS 18 “doesn’t exist” as of its cutoff.

**Outdated?**
- Yes. It refers to a 2023 cutoff even though iOS 18 shipped in September 2024.

**Confident but Wrong?**
- Yes. It asserts as if fact that “iOS 18” does not exist, which is incorrect in late 2024.

**Does it say “I don’t know”?**
 - No—the model simply claims “there is no iOS 18,” rather than explicitly admitting ignorance.

# **Part 1: Load a Custom PDF Knowledge Base**

Find blog posts or wikipedia page with your topic and save information about it to a PDF file, and load it using `PyPDFLoader`. You may use other loaders not only pdf, but pdf loader is exactly the same as we used during lab.

- Find informative content on your topic (Wikipedia page, blog post, article, etc.)
- Save the page as a PDF file (you can use your browser’s print-to-PDF feature)
"""

# Define the path to the PDF file containing the knowledge base
file_path = "IOS_18.pdf"

# Create a PyPDFLoader object to handle loading of the PDF
loader = PyPDFLoader(file_path)

# Load the contents of the PDF into a variable for further processing
RAW_KNOWLEDGE_BASE = loader.load()

print(f"Number of pages loaded: {len(RAW_KNOWLEDGE_BASE)}")

first_page = RAW_KNOWLEDGE_BASE[0].page_content
print("----- First page excerpt -----")
print(first_page[:500], "...\n")

"""
# **Part 2: Repeat the Lab with Your Own Knowledge Base + RAG Tuning**

## **Goal:**

Practice building a **RAG pipeline** and explore how **chunk size** and **chunk overlap** affect the quality of LLM answers to different questions.

---

## **What You Need to Do:**

1. **Repeat the Lab Using Your PDF Knowledge Base**

   * Use the PDF file you selected and loaded in Part 1.

2. **Create 3 Different Questions**

   * Design **three meaningful, specific questions** based on your topic.
   * Each question must be clearly related to the content of your PDF.

3. **Run RAG for Each Question with 3 Different Settings:**
   For each question:

   * Run the RAG pipeline **three times** using different settings for:

     * `chunk_size` (e.g., 100, 300, 500)
     * `chunk_overlap` (e.g., 0, 20, 50, 100)
   * This means you will run a total of **9 tests** (3 questions × 3 settings each).


4. **Answer Each Question Using an LLM**

   * Use the loaded chunks and a retriever to find relevant parts.
   * Pass the retrieved context to the LLM and generate an answer.
   * You can use similar tools as we used in the Lab

5. **Explain Your Results**
   For each of the 3 questions:

   * Write a short **description of the question** and **why you chose it**.
   * **Compare the answers** you got using different settings.
   * Reflect on:

     * How answer quality changed with different `chunk_size` and `chunk_overlap`
     * Which setting gave the most useful or accurate result
     * Why you think it performed better/worse

---
"""

def split_documents(chunk_size: int, chunk_overlap: int, knowledge_base: List, tokenizer_name: str) -> List:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs_processed = text_splitter.split_documents(knowledge_base)

    return docs_processed

def separator(title="", width=80, char="="):
    print(char * width)
    if title:
        print(title.center(width))
        print(char * width)

template_system = (
    "Using the information contained in the context, give a comprehensive answer to the question. "
    "Respond only to the question asked; response should be concise and relevant to the question. "
    "If the answer cannot be deduced from the context, respond with exactly 'Information not available.'"
)
template_user = "Context: {context}\n---\nNow here is the question you need to answer.\nQuestion: {question}"
prompt_in_chat_format = [
    {"role": "system", "content": template_system},
    {"role": "user", "content": template_user}
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format,
    tokenize=False,
    add_generation_prompt=True
)



custom_questions = [
    {
        "question": "What are the main new features introduced in iOS 18?",
        "reason": "Identifies primary features to understand iOS 18's functional improvements."
    },
    {
        "question": "Describe the major user-interface changes in iOS 18.",
        "reason": "Focuses on UI to assess design and usability enhancements."
    },
    {
        "question": "Which security improvements did Apple add in iOS 18?",
        "reason": "Security is critical; this question evaluates iOS 18's safety advancements."
    }
]


settings = [
    {"chunk_size": 200, "chunk_overlap": 40},
    {"chunk_size": 400, "chunk_overlap": 80},
    {"chunk_size": 800, "chunk_overlap": 160}
]

def get_embedding_model(device):
    return HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

device_name = "cuda" if torch.cuda.is_available() else "cpu"

report = {
    "topic": "iOS 18",
    "question": initial_question,
    "answer": base_answer,
    "rag": []
}

for cq in custom_questions:
    q_text = cq["question"]
    q_reason = cq["reason"]
    question_entry = {"question": q_text, "reason": q_reason, "experiments": []}
    separator(f"RAG QUESTION: {q_text}")

    for setting in settings:
        chunk_size = setting["chunk_size"]
        chunk_overlap = setting["chunk_overlap"]
        sep_title = f"Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        separator(sep_title, width=80, char="-")

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs_chunks = splitter.split_documents(RAW_KNOWLEDGE_BASE)

        persist_dir = f"chroma_db_{chunk_size}_{chunk_overlap}"
        embedding_model = get_embedding_model(device_name)
        if not os.path.exists(persist_dir):
            vector_store = Chroma.from_documents(
                docs_chunks,
                embedding_model,
                persist_directory=persist_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )
        else:
            vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_model.embed_query,
                collection_metadata={"hnsw:space": "cosine"},
            )

        query_embedding = embedding_model.embed_query(q_text)
        results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=4)

        retrieved_docs_text = [doc.page_content for doc in results]
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {i}:::\n{doc}\n" for i, doc in enumerate(retrieved_docs_text)])

        final_prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=q_text)

        rag_output = READER_LLM(final_prompt)
        rag_answer = rag_output[0]["generated_text"].strip() if (rag_output := READER_LLM(final_prompt)) else ""
        if "Information not available" in rag_answer or not rag_answer:
            rag_answer = "Information not available."

        print("Answer:")
        print(textwrap.fill(rag_answer, width=80))
        print()

        reflection_text = (
            f"For chunk_size={chunk_size} and chunk_overlap={chunk_overlap}, the answer started '"
            f"{rag_answer[:50]}...' Evaluate completeness and accuracy compared to other settings."
        )

        experiment_entry = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "answer": rag_answer,
            "reflection": reflection_text
        }
        question_entry["experiments"].append(experiment_entry)

    report["rag"].append(question_entry)
    print()

output_filename = "rag_report_hosseini_sobhan_2141890.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)


import json

with open("rag_report_Arkadiusz_Modzelewski_29580.json", "w", encoding="utf-8") as f:
    json.dump(your_results_dict, f, indent=2, ensure_ascii=False)
from src.retrieval.retriever import RetrieverConfig, build_retriever, retrieve
from llm_client import LLMConfig, build_llm_layer, answer

retriever = build_retriever(RetrieverConfig())
llm  = build_llm_layer(LLMConfig())


def run_pipeline_on_dataset(dataset: list[dict]) -> list[dict]:
    """
    Run the full RAG pipeline on every eval question and
    collect inputs/outputs needed by RAGAS.
    """
    results = []

    for item in dataset:
        question = item["question"]

        # Retrieve
        chunks = retrieve(question, retriever, top_n=5)
        contexts = [c.content for c in chunks]

        # Generate
        response = answer(question, chunks, llm)

        results.append({
            "question": question,
            "answer": response.answer,
            "contexts": contexts,        # list[str] — chunks fed to LLM
            "ground_truth": item["ground_truth"],
        })

        print(f"✓ {question[:60]}")

    return results
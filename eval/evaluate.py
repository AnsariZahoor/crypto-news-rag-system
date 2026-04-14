import os

from dotenv import load_dotenv 

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness, FactualCorrectness
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper

from eval.run_eval import run_pipeline_on_dataset
from eval.dataset import EVAL_DATASET


load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

def evaluate_pipeline():
    # 1. Run pipeline on all eval questions
    results = run_pipeline_on_dataset(EVAL_DATASET)

    # 2. Convert to HuggingFace Dataset (RAGAS format)
    hf_dataset = Dataset.from_list(results)

    # 3. Use same LLM + embedder you use in production
    ragas_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0))

    ragas_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    # 4. Evaluate
    scores = evaluate(
        dataset = hf_dataset,
        metrics=[ContextPrecision(), ContextRecall(), Faithfulness(), FactualCorrectness()],
        llm = ragas_llm,
        embeddings = ragas_embeddings,
    )

    return scores


if __name__ == "__main__":
    scores = evaluate_pipeline()
    df = scores.to_pandas()
    print(df)

    df.to_csv("results.csv", index=False)
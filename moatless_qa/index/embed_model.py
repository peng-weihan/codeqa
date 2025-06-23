import os  
from dotenv import load_dotenv

load_dotenv()

from moatless_qa.index.retry_voyage_embedding import VoyageEmbeddingWithRetry


# def get_embed_model(model_name: str) -> "BaseEmbedding":
#     if model_name.startswith("voyage"):
#         try:
#             from llama_index.embeddings.voyageai import VoyageEmbedding
#         except ImportError as e:
#             raise ImportError(
#                 "llama-index-embeddings-voyageai is not installed. Please install it using `pip install llama-index-embeddings-voyageai`"
#             ) from e

#         if "VOYAGE_API_KEY" not in os.environ:
#             raise ValueError(
#                 "VOYAGE_API_KEY environment variable is not set. Please set it to your Voyage API key."
#             )
#         return VoyageEmbeddingWithRetry(
#             model_name=model_name,
#             voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
#             truncation=True,
#             embed_batch_size=80,
#         )
#     else:
#         # Assumes OpenAI otherwise
#         try:
#             from llama_index.embeddings.openai import OpenAIEmbedding
#         except ImportError as e:
#             raise ImportError(
#                 "llama-index-embeddings-openai is not installed. Please install it using `pip install llama-index-embeddings-openai`"
#             ) from e

#         return OpenAIEmbedding(model_name=model_name)
    

def get_embed_model(model_name: str) -> "BaseEmbedding":

    try:
        from llama_index.embeddings.voyageai import VoyageEmbedding
    except ImportError as e:
        raise ImportError(
            "llama-index-embeddings-voyageai is not installed. Please install it using `pip install llama-index-embeddings-voyageai`"
        ) from e

    if "VOYAGE_API_KEY" not in os.environ:
        raise ValueError(
            "VOYAGE_API_KEY environment variable is not set. Please set it to your Voyage API key."
        )
    return VoyageEmbeddingWithRetry(
        model_name="voyage-3",
        voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        truncation=True,
        embed_batch_size=80,
    )

# def get_embed_model(model_name: str) -> "BaseEmbedding":
#     try:
#         from sentence_transformers import SentenceTransformer
#     except ImportError:
#         raise ImportError(
#             "请先安装 sentence-transformers 和 llama-index 的相关依赖: pip install sentence-transformers llama-index"
#         )
#     # model_name 示例：'st-all-MiniLM-L6-v2' 或 'sentence-transformers/all-MiniLM-L6-v2'
#     return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
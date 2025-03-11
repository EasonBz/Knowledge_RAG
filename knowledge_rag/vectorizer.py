"""向量化引擎模块

这个模块负责将文本内容转换为向量表示，用于语义检索。
使用预训练的语言模型将文本转换为高维向量空间中的点。
"""

from typing import List, Dict, Any, Union
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document


class TextVectorizer:
    """文本向量化器，负责将文本转换为向量表示"""
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 
                 use_openai: bool = False,
                 openai_api_key: str = None,
                 openai_model: str = "text-embedding-ada-002"):
        """初始化文本向量化器
        
        Args:
            model_name: 使用的Hugging Face模型名称
            use_openai: 是否使用OpenAI的嵌入模型
            openai_api_key: OpenAI API密钥
            openai_model: OpenAI嵌入模型名称
        """
        self.model_name = model_name
        self.use_openai = use_openai
        
        if use_openai:
            if not openai_api_key:
                raise ValueError("使用OpenAI嵌入时必须提供API密钥")
            self.embeddings = OpenAIEmbeddings(
                model=openai_model,
                openai_api_key=openai_api_key
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'}
            )
    
    def vectorize_text(self, text: str) -> List[float]:
        """将单个文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            文本的向量表示
        """
        return self.embeddings.embed_query(text)
    
    def vectorize_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """将文档列表转换为向量
        
        Args:
            documents: Document对象列表
            
        Returns:
            包含文档ID、向量和元数据的字典
        """
        result = {
            "ids": [],
            "embeddings": [],
            "metadatas": [],
            "documents": []
        }
        
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            embedding = self.vectorize_text(doc.page_content)
            
            result["ids"].append(doc_id)
            result["embeddings"].append(embedding)
            result["metadatas"].append(doc.metadata)
            result["documents"].append(doc.page_content)
        
        return result
    
    def vectorize_query(self, query: str) -> List[float]:
        """将查询文本转换为向量
        
        Args:
            query: 查询文本
            
        Returns:
            查询的向量表示
        """
        return self.vectorize_text(query)
"""检索引擎模块

这个模块负责根据查询找到最相关的知识片段。
使用向量存储来检索相关文档，并计算相关性分数。
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np

from langchain_core.documents import Document  # 更新导入
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from knowledge_rag.vector_store import VectorStore


class KnowledgeRetriever(BaseRetriever):
    """知识检索器，负责检索相关文档"""
    
    # 定义 Pydantic 字段
    vector_store: VectorStore = Field(description="向量存储")
    top_k: int = Field(default=4, description="返回的文档数量")
    score_threshold: float = Field(default=0.0, description="相似度分数阈值，低于此分数的文档将被过滤")
    
    def __init__(self, vector_store: VectorStore, top_k: int = 4, score_threshold: float = 0.0):
        """初始化知识检索器
        
        Args:
            vector_store: 向量存储
            top_k: 返回的文档数量
            score_threshold: 相似度分数阈值，低于此分数的文档将被过滤
        """
        # 使用父类的初始化方法，传入字段值
        super().__init__(vector_store=vector_store, top_k=top_k, score_threshold=score_threshold)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """获取与查询相关的文档
        
        Args:
            query: 查询文本
            
        Returns:
            相关文档列表
        """
        return self.vector_store.get_relevant_documents(query, k=self.top_k)
    
    def get_relevant_documents_with_score(self, query: str) -> List[tuple]:
        """获取与查询相关的文档及其相似度分数
        
        Args:
            query: 查询文本
            
        Returns:
            (文档, 相似度分数)元组列表
        """
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        
        # 如果设置了分数阈值，则过滤低分文档
        if self.score_threshold > 0:
            results = [(doc, score) for doc, score in results if score >= self.score_threshold]
        
        return results
    
    def retrieve_with_metadata(self, query: str) -> Dict[str, Any]:
        """检索相关文档并返回带有元数据的结果
        
        Args:
            query: 查询文本
            
        Returns:
            包含文档、分数和元数据的字典
        """
        results = self.get_relevant_documents_with_score(query)
        
        return {
            "documents": [doc for doc, _ in results],
            "scores": [score for _, score in results],
            "metadatas": [doc.metadata for doc, _ in results],
            "query": query
        }
    
    def extract_related_concepts(self, documents: List[Document], top_n: int = 5) -> List[str]:
        """从相关文档中提取关键概念
        
        Args:
            documents: 文档列表
            top_n: 返回的概念数量
            
        Returns:
            关键概念列表
        """
        # 简单实现：提取文档中最常见的名词短语
        # 在实际应用中，可以使用更复杂的方法，如关键词提取算法
        concepts = set()
        for doc in documents:
            # 这里可以添加更复杂的概念提取逻辑
            # 例如使用NLTK或spaCy提取名词短语
            # 简单起见，这里只是分割文本并取前几个词
            words = doc.page_content.split()[:10]
            for word in words:
                if len(word) > 3:  # 简单过滤短词
                    concepts.add(word)
        
        return list(concepts)[:top_n]
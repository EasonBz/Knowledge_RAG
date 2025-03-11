"""向量存储模块

这个模块负责存储文档的向量表示和元数据，并提供检索功能。
支持使用FAISS和Chroma作为向量存储的后端。
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np

from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document  # 添加这行导入

from knowledge_rag.vectorizer import TextVectorizer


class VectorStore:
    """向量存储，负责存储和检索文档向量"""
    
    def __init__(self, embeddings: Optional[Embeddings] = None, 
                 persist_directory: Optional[str] = None,
                 store_type: str = "faiss"):
        """初始化向量存储
        
        Args:
            embeddings: 嵌入模型
            persist_directory: 持久化目录
            store_type: 存储类型，支持"faiss"和"chroma"
        """
        self.embeddings = embeddings or TextVectorizer()
        self.persist_directory = persist_directory
        self.store_type = store_type.lower()
        self.vector_store = None
        
        if self.persist_directory and os.path.exists(self.persist_directory):
            self._load_vector_store()
    
    def _load_vector_store(self):
        """从持久化目录加载向量存储"""
        if not self.persist_directory or not os.path.exists(self.persist_directory):
            return
        
        if self.store_type == "faiss":
            self.vector_store = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        elif self.store_type == "chroma":
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            raise ValueError(f"不支持的存储类型: {self.store_type}")
    
    def add_documents(self, documents: List[Document]):
        """添加文档到向量存储
        
        Args:
            documents: Document对象列表
        """
        if not documents:
            return
        
        if self.vector_store is None:
            if self.store_type == "faiss":
                self.vector_store = FAISS.from_documents(
                    documents, self.embeddings
                )
            elif self.store_type == "chroma":
                self.vector_store = Chroma.from_documents(
                    documents,
                    self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                raise ValueError(f"不支持的存储类型: {self.store_type}")
        else:
            self.vector_store.add_documents(documents)
        
        # 如果设置了持久化目录，则保存向量存储
        if self.persist_directory:
            self._persist()
    
    def _persist(self):
        """持久化向量存储"""
        if not self.persist_directory:
            return
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        if self.store_type == "faiss":
            self.vector_store.save_local(self.persist_directory)
        elif self.store_type == "chroma" and hasattr(self.vector_store, "persist"):
            self.vector_store.persist()
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """相似性搜索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            相似文档列表
        """
        if self.vector_store is None:
            return []
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """带分数的相似性搜索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            (文档, 相似度分数)元组列表
        """
        if self.vector_store is None:
            return []
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """获取相关文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            相关文档列表
        """
        return self.similarity_search(query, k=k)
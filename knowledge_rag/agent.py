"""生物学知识Agent接口

这个模块实现了生物学知识Agent的主要接口，
负责接收查询、执行检索并生成结构化响应。
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import numpy as np

from langchain.schema import Document

from knowledge_rag.document_loader import load_documents
from knowledge_rag.vectorizer import TextVectorizer
from knowledge_rag.vector_store import VectorStore
from knowledge_rag.retriever import KnowledgeRetriever


class BioKnowledgeAgent:
    """生物学知识Agent，负责处理生物学知识查询"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 use_openai: bool = False,
                 openai_api_key: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 store_type: str = "faiss",
                 top_k: int = 4,
                 score_threshold: float = 0.0):
        """初始化生物学知识Agent
        
        Args:
            model_name: 使用的Hugging Face模型名称
            use_openai: 是否使用OpenAI的嵌入模型
            openai_api_key: OpenAI API密钥
            persist_directory: 向量存储持久化目录
            store_type: 向量存储类型，支持"faiss"和"chroma"
            top_k: 检索返回的文档数量
            score_threshold: 相似度分数阈值
        """
        # 初始化向量化引擎
        self.vectorizer = TextVectorizer(
            model_name=model_name,
            use_openai=use_openai,
            openai_api_key=openai_api_key
        )
        
        # 初始化向量存储
        self.vector_store = VectorStore(
            embeddings=self.vectorizer.embeddings,
            persist_directory=persist_directory,
            store_type=store_type
        )
        
        # 初始化检索引擎
        self.retriever = KnowledgeRetriever(
            vector_store=self.vector_store,
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        # 已加载的文档源
        self.document_sources = set()
    
    def ingest_documents(self, documents: List[Document]):
        """摄入文档到知识库
        
        Args:
            documents: Document对象列表
        """
        if not documents:
            return
        
        # 添加文档到向量存储
        self.vector_store.add_documents(documents)
        
        # 记录文档源
        for doc in documents:
            if "source" in doc.metadata:
                self.document_sources.add(doc.metadata["source"])
    
    def ingest_from_source(self, source: Union[str, List[str]], chunk_size: int = 1000, chunk_overlap: int = 200):
        """从源加载并摄入文档
        
        Args:
            source: 文档路径或路径列表
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        documents = load_documents(source, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.ingest_documents(documents)
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """查询知识库
        
        Args:
            query_text: 查询文本
            
        Returns:
            结构化响应，包含答案、来源、置信度和相关概念
        """
        # 检索相关文档
        retrieval_result = self.retriever.retrieve_with_metadata(query_text)
        documents = retrieval_result["documents"]
        scores = retrieval_result["scores"]
        metadatas = retrieval_result["metadatas"]
        
        if not documents:
            return {
                "answer": "抱歉，我没有找到相关信息。",
                "sources": [],
                "confidence": 0.0,
                "related_concepts": []
            }
        
        # 提取相关概念
        related_concepts = self.retriever.extract_related_concepts(documents)
        
        # 计算置信度（简单实现：使用最高相似度分数）
        confidence = float(max(scores)) if scores else 0.0
        
        # 提取来源
        sources = []
        for metadata in metadatas:
            if "source" in metadata and metadata["source"] not in sources:
                sources.append(metadata["source"])
        
        # 构建响应
        # 注意：在实际应用中，这里可以使用LLM来生成答案
        # 简单起见，这里只是拼接最相关的文档内容
        answer = "\n\n".join([doc.page_content for doc in documents[:2]])
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "related_concepts": related_concepts
        }
    
    def query_with_knowledge_graph(self, query_text: str) -> Dict[str, Any]:
        """查询知识库并生成知识图谱
        
        Args:
            query_text: 查询文本
            
        Returns:
            结构化响应，包含答案、来源、置信度、相关概念和知识图谱
        """
        # 获取基本响应
        response = self.query(query_text)
        
        # 添加知识图谱
        # 注意：在实际应用中，这里可以使用更复杂的方法来构建知识图谱
        # 简单起见，这里只是创建一个简单的实体-关系结构
        entities = []
        relationships = []
        
        # 从相关概念中提取实体
        for i, concept in enumerate(response["related_concepts"]):
            entities.append({
                "id": f"entity_{i}",
                "name": concept,
                "type": "concept"
            })
        
        # 添加查询作为中心实体
        entities.append({
            "id": "query",
            "name": query_text,
            "type": "query"
        })
        
        # 创建查询与概念之间的关系
        for i in range(len(response["related_concepts"])):
            relationships.append({
                "source": "query",
                "target": f"entity_{i}",
                "type": "related_to"
            })
        
        # 添加知识图谱到响应
        response["knowledge_graph"] = {
            "entities": entities,
            "relationships": relationships
        }
        
        return response
    
    def get_document_sources(self) -> List[str]:
        """获取已加载的文档源
        
        Returns:
            文档源列表
        """
        return list(self.document_sources)
    
    def save(self, directory: str):
        """保存Agent状态
        
        Args:
            directory: 保存目录
        """
        os.makedirs(directory, exist_ok=True)
        
        # 保存向量存储
        vector_store_dir = os.path.join(directory, "vector_store")
        os.makedirs(vector_store_dir, exist_ok=True)
        
        # 更新向量存储的持久化目录并保存
        self.vector_store.persist_directory = vector_store_dir
        self.vector_store._persist()
        
        # 保存文档源信息
        with open(os.path.join(directory, "sources.json"), "w") as f:
            json.dump(list(self.document_sources), f)
    
    @classmethod
    def load(cls, directory: str, **kwargs):
        """加载Agent状态
        
        Args:
            directory: 保存目录
            **kwargs: 其他参数
            
        Returns:
            BioKnowledgeAgent实例
        """
        vector_store_dir = os.path.join(directory, "vector_store")
        
        # 创建Agent实例，指定向量存储目录
        agent = cls(persist_directory=vector_store_dir, **kwargs)
        
        # 加载文档源信息
        sources_file = os.path.join(directory, "sources.json")
        if os.path.exists(sources_file):
            with open(sources_file, "r") as f:
                agent.document_sources = set(json.load(f))
        
        return agent
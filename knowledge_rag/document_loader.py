"""
文档加载和处理模块
这个模块负责加载各种格式的生物学文献（PDF、DOCX等），
并将它们处理成适合向量化和检索的文本块。
"""

import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 这个导入暂时不需要更改
from langchain_core.documents import Document  # 已更新


class DocumentProcessor:
    """文档处理器，负责加载和分割文档"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """初始化文档处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档
        
        Args:
            file_path: 文档路径
            
        Returns:
            Document对象列表
        """
        file_path = os.path.abspath(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        elif file_ext in [".txt", ".md"]:
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_ext}")
        
        documents = loader.load()
        
        # 添加元数据
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["source"] = file_path
            doc.metadata["file_type"] = file_ext
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成更小的文本块
        
        Args:
            documents: Document对象列表
            
        Returns:
            分割后的Document对象列表
        """
        return self.text_splitter.split_documents(documents)
    
    def process_document(self, file_path: str) -> List[Document]:
        """处理单个文档：加载并分割
        
        Args:
            file_path: 文档路径
            
        Returns:
            处理后的Document对象列表
        """
        documents = self.load_document(file_path)
        return self.split_documents(documents)
    
    def process_directory(self, dir_path: str, glob_pattern: str = "**/*.*") -> List[Document]:
        """处理目录中的所有文档
        
        Args:
            dir_path: 目录路径
            glob_pattern: 文件匹配模式
            
        Returns:
            处理后的Document对象列表
        """
        dir_path = os.path.abspath(dir_path)
        loader = DirectoryLoader(
            dir_path,
            glob=glob_pattern,
            loader_cls=lambda file_path: self._get_loader_for_file(file_path)
        )
        documents = loader.load()
        return self.split_documents(documents)
    
    def _get_loader_for_file(self, file_path: str):
        """根据文件类型获取合适的加载器
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档加载器
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == ".pdf":
            return PyPDFLoader(file_path)
        elif file_ext in [".docx", ".doc"]:
            return Docx2txtLoader(file_path)
        elif file_ext in [".txt", ".md"]:
            return TextLoader(file_path)
        else:
            # 跳过不支持的文件类型
            return None


def load_documents(source: Union[str, List[str]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """加载文档的便捷函数
    
    Args:
        source: 文档路径或路径列表
        chunk_size: 文本块大小
        chunk_overlap: 文本块重叠大小
        
    Returns:
        处理后的Document对象列表
    """
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    if isinstance(source, str):
        path = Path(source)
        if path.is_dir():
            return processor.process_directory(str(path))
        else:
            return processor.process_document(str(path))
    elif isinstance(source, list):
        documents = []
        for path in source:
            path_obj = Path(path)
            if path_obj.is_dir():
                documents.extend(processor.process_directory(str(path_obj)))
            else:
                documents.extend(processor.process_document(str(path_obj)))
        return documents
    else:
        raise ValueError("source必须是字符串路径或路径列表")
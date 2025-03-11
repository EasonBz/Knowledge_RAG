# 生物学知识RAG系统

一个基于检索增强生成（RAG）架构的生物学知识Agent，能够从生物学科文献和细胞数据库中提取知识，并以结构化的形式提供给主干LLM使用。

## 项目架构

由以下几个主要组件构成：

1. **文档处理器**：负责加载、解析和分割生物学论文，提取关键信息。
2. **向量化引擎**：将文本内容转换为向量表示，用于语义检索。
3. **向量存储**：存储文档的向量表示和元数据。
4. **检索引擎**：根据查询找到最相关的知识片段。
5. **Agent接口**：接收查询，执行检索，生成结构化响应。

文档首先通过document_loader加载，然后由vectorizer转换为向量，存入vector_store，当用户查询时，retriever从vector_store中检索相关信息，最后由agent整合结果并返回结构化响应。每个模块都有特定的输入输出接口，确保数据能够顺畅地在整个系统中流动。

## 使用方法

```python
# 加载文档
from knowledge_rag.document_loader import load_documents
from knowledge_rag.agent import BioKnowledgeAgent

# 加载文档并创建Agent
docs = load_documents("path/to/papers/")
agent = BioKnowledgeAgent()
agent.ingest_documents(docs)

# 查询知识
response = agent.query("什么是CRISPR-Cas9系统？")
print(response)
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 项目结构

```
knowledge_rag/
├── document_loader.py  # 文档加载和处理
├── vectorizer.py       # 文本向量化
├── vector_store.py     # 向量存储
├── retriever.py        # 知识检索
├── agent.py            # Agent接口
└── utils.py            # 工具函数

tests/                  # 测试用例
```

## 输出格式

Agent的输出采用结构化JSON格式，包含以下字段：

```json
{
  "answer": "对查询的直接回答",
  "sources": ["引用的文献来源"],
  "confidence": 0.95,  // 置信度分数
  "related_concepts": ["相关概念列表"],
  "knowledge_graph": {  // 可选的知识图谱表示
    "entities": [...],
    "relationships": [...]
  }
}
```

这种结构化输出使主干LLM能够轻松整合这些信息进行进一步推理。
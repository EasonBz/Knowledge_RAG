"""测试生物学知识Agent

这个模块包含对BioKnowledgeAgent的测试用例，
验证其文档加载、知识检索和响应生成功能。
"""

import os
import unittest
from pathlib import Path

from knowledge_rag.agent import BioKnowledgeAgent
from knowledge_rag.document_loader import load_documents


class TestBioKnowledgeAgent(unittest.TestCase):
    """测试BioKnowledgeAgent的功能"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试目录
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # 创建测试文档
        self.test_doc_path = self.test_dir / "test_doc.txt"
        with open(self.test_doc_path, "w") as f:
            f.write("""
            CRISPR-Cas9是一种革命性的基因编辑技术，它允许科学家精确地修改DNA序列。
            CRISPR代表"成簇规律间隔短回文重复序列"，而Cas9是一种与CRISPR系统相关的酶。
            这项技术源于细菌的免疫系统，细菌使用CRISPR-Cas系统来识别和切割入侵的病毒DNA。
            
            在基因编辑应用中，CRISPR-Cas9系统包含两个关键组件：
            1. 向导RNA (gRNA)：这是一小段RNA，设计用来与目标DNA序列互补配对。
            2. Cas9酶：这是一种能够切割DNA的蛋白质。
            
            当gRNA引导Cas9到达目标DNA序列时，Cas9会在特定位置切割DNA，允许删除、添加或替换基因。
            这项技术的应用范围广泛，包括基础研究、医学治疗和农业改良。
            在医学领域，CRISPR-Cas9有望治疗遗传性疾病，如镰状细胞贫血和囊性纤维化。
            """)
        
        # 创建持久化目录
        self.persist_dir = self.test_dir / "vector_store"
        
        # 初始化Agent
        self.agent = BioKnowledgeAgent(
            persist_directory=str(self.persist_dir),
            store_type="faiss"
        )
        
        # 加载测试文档
        self.agent.ingest_from_source(str(self.test_doc_path))
    
    def tearDown(self):
        """测试后的清理工作"""
        '''
        # 删除测试文件
        if self.test_doc_path.exists():
            self.test_doc_path.unlink()
        
        # 递归删除测试目录
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            '''
    
    def test_document_loading(self):
        """测试文档加载功能"""
        # 验证文档源是否正确记录
        sources = self.agent.get_document_sources()
        self.assertEqual(len(sources), 1)
        self.assertTrue(str(self.test_doc_path.absolute()) in sources)
    
    def test_query(self):
        """测试知识查询功能"""
        # 执行查询
        query = "什么是CRISPR-Cas9系统？"
        response = self.agent.query(query)
        
        # 验证响应结构
        self.assertIn("answer", response)
        self.assertIn("sources", response)
        self.assertIn("confidence", response)
        self.assertIn("related_concepts", response)
        
        # 验证响应内容
        self.assertGreater(len(response["answer"]), 0)
        self.assertEqual(len(response["sources"]), 1)
        self.assertGreaterEqual(response["confidence"], 0.0)
    
    def test_query_with_knowledge_graph(self):
        """测试带知识图谱的查询功能"""
        # 执行查询
        query = "CRISPR-Cas9的应用是什么？"
        response = self.agent.query_with_knowledge_graph(query)
        
        # 验证响应结构
        self.assertIn("knowledge_graph", response)
        self.assertIn("entities", response["knowledge_graph"])
        self.assertIn("relationships", response["knowledge_graph"])
        
        # 验证知识图谱内容
        entities = response["knowledge_graph"]["entities"]
        relationships = response["knowledge_graph"]["relationships"]
        
        # 至少应该有查询实体和一些概念实体
        self.assertGreaterEqual(len(entities), 1)
        self.assertGreaterEqual(len(relationships), 0)
    
    def test_save_and_load(self):
        """测试Agent状态的保存和加载"""
        # 保存Agent状态
        save_dir = self.test_dir / "saved_agent"
        self.agent.save(str(save_dir))
        
        # 验证保存的文件是否存在
        self.assertTrue((save_dir / "sources.json").exists())
        self.assertTrue((save_dir / "vector_store").exists())
        
        # 加载Agent状态
        loaded_agent = BioKnowledgeAgent.load(str(save_dir))
        
        # 验证加载的Agent
        sources = loaded_agent.get_document_sources()
        self.assertEqual(len(sources), 1)
        self.assertTrue(str(self.test_doc_path.absolute()) in sources)
        
        # 使用加载的Agent执行查询
        query = "CRISPR-Cas9是什么？"
        response = loaded_agent.query(query)
        
        # 验证响应
        self.assertIn("answer", response)
        self.assertGreater(len(response["answer"]), 0)


if __name__ == "__main__":
    unittest.main()
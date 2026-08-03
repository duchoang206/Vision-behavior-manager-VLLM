import os
from typing import Optional

class RAGManager:
    def __init__(self, db_path="./data/chroma", collection_name="vms_docs"):
        self.db_path = db_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._initialized = False

    def _ensure_init(self):
        if self._initialized:
            return
        try:
            import chromadb
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            self._initialized = True
        except Exception as e:
            print(f"[RAG] ChromaDB init error: {e}")

    def initialize_with_pdf(self, pdf_path):
        try:
            self._ensure_init()
            if not self._collection:
                return

            if self._collection.count() > 0:
                print(f"[RAG] Database đã có {self._collection.count()} chunks. Bỏ qua trích xuất PDF.")
                return

            if not os.path.exists(pdf_path):
                print(f"[RAG] Không tìm thấy file PDF tại {pdf_path}")
                return

            from pypdf import PdfReader
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            self._collection.add(documents=chunks, ids=ids)
            print(f"[RAG] Đã vector hoá và lưu {len(chunks)} chunks vào ChromaDB.")
        except Exception as e:
            print(f"[RAG] initialize_with_pdf error: {e}")

    def query_rag(self, query: str, top_k: int = 2):
        try:
            self._ensure_init()
            if not self._collection or self._collection.count() == 0:
                return "Không có dữ liệu tài liệu kỹ thuật nào được tìm thấy."

            results = self._collection.query(
                query_texts=[query],
                n_results=top_k
            )

            if not results.get('documents') or len(results['documents'][0]) == 0:
                return "Không tìm thấy thông tin tương ứng trong tài liệu kỹ thuật."

            context_chunks = results['documents'][0]
            return "\n\n".join(context_chunks)
        except Exception as e:
            return f"Không thể tra cứu RAG: {e}"

# Khởi tạo lazy instance an toàn
db_path = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
rag_engine = RAGManager(db_path=db_path)


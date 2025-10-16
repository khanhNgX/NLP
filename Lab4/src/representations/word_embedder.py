"""
Lop WordEmbedder: Tai va su dung mo hinh word embedding da duoc huan luyen truoc.
"""
import gensim.downloader as api
import numpy as np
from typing import List, Tuple
from src.preprocessing.simple_tokenizer import SimpleTokenizer


class WordEmbedder:
    """
    Lop de tai va su dung cac mo hinh word embedding da duoc huan luyen truoc.
    """
    
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        """
        Khoi tao WordEmbedder voi mo hinh da duoc huan luyen truoc.
        
        Args:
            model_name: Ten cua mo hinh trong kho du lieu gensim
                       (vi du: 'glove-wiki-gigaword-50')
        """
        print(f"Dang tai mo hinh '{model_name}'...")
        print("Lan dau tien tai co the mat vai phut de tai xuong (~65MB)...")
        self.model = api.load(model_name)
        self.model_name = model_name
        self.vector_size = self.model.vector_size
        self.tokenizer = SimpleTokenizer()
        print(f"Da tai xong mo hinh! Kich thuoc vector: {self.vector_size}")
    
    def get_vector(self, word: str) -> np.ndarray:
        """
        Lay vector embedding cho mot tu.
        
        Args:
            word: Tu can lay vector
            
        Returns:
            Vector numpy. Neu tu khong co trong tu vung, tra ve vector 0.
        """
        try:
            return self.model[word.lower()]
        except KeyError:
            print(f"Canh bao: Tu '{word}' khong co trong tu vung (OOV)")
            return np.zeros(self.vector_size)
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """
        Tinh do tuong dong cosine giua hai tu.
        
        Args:
            word1: Tu thu nhat
            word2: Tu thu hai
            
        Returns:
            Do tuong dong cosine (tu -1 den 1). Gia tri cao hon = tuong dong hon.
            Tra ve 0.0 neu mot trong cac tu khong co trong tu vung.
        """
        try:
            return self.model.similarity(word1.lower(), word2.lower())
        except KeyError as e:
            print(f"Canh bao: Mot hoac ca hai tu khong co trong tu vung: {e}")
            return 0.0
    
    def get_most_similar(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Tim cac tu tuong dong nhat voi tu cho truoc.
        
        Args:
            word: Tu can tim cac tu tuong dong
            top_n: So luong tu tuong dong nhat can tra ve
            
        Returns:
            Danh sach cac cap (tu, do_tuong_dong) duoc sap xep theo do tuong dong giam dan.
            Tra ve danh sach rong neu tu khong co trong tu vung.
        """
        try:
            return self.model.most_similar(word.lower(), topn=top_n)
        except KeyError:
            print(f"Canh bao: Tu '{word}' khong co trong tu vung")
            return []
    
    def embed_document(self, document: str) -> np.ndarray:
        """
        Tao embedding cho mot tai lieu bang cach tinh trung binh cac vector tu.
        
        Chien luoc: Tach tai lieu thanh cac tu, lay vector cho moi tu,
        roi tinh trung binh cong cua tat ca cac vector.
        
        Args:
            document: Van ban tai lieu can embedding
            
        Returns:
            Vector embedding cua tai lieu (trung binh cac vector tu).
            Tra ve vector 0 neu tai lieu khong chua tu nao trong tu vung.
        """
        # Tach tai lieu thanh cac tu
        tokens = self.tokenizer.tokenize(document)
        
        # Lay vector cho moi tu (bo qua cac tu OOV)
        vectors = []
        for token in tokens:
            try:
                vec = self.model[token]
                vectors.append(vec)
            except KeyError:
                # Bo qua cac tu khong co trong tu vung
                continue
        
        # Neu khong co tu nao duoc tim thay, tra ve vector 0
        if len(vectors) == 0:
            print(f"Canh bao: Khong co tu nao trong tai lieu duoc tim thay trong tu vung")
            return np.zeros(self.vector_size)
        
        # Tinh trung binh cac vector
        document_vector = np.mean(vectors, axis=0)
        return document_vector
    
    def get_vocabulary_size(self) -> int:
        """
        Lay kich thuoc tu vung cua mo hinh.
        
        Returns:
            So luong tu trong tu vung
        """
        return len(self.model)
    
    def word_in_vocabulary(self, word: str) -> bool:
        """
        Kiem tra xem mot tu co trong tu vung hay khong.
        
        Args:
            word: Tu can kiem tra
            
        Returns:
            True neu tu co trong tu vung, False neu khong
        """
        return word.lower() in self.model

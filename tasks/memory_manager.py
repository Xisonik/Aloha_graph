import torch
import torchvision.transforms as transforms
import clip
from PIL import Image
from collections import deque
import numpy as np
from torchvision.utils import save_image, make_grid

class ImageMemoryManager:
    def __init__(self, capacity=4, device="cuda", similarity_threshold=0.9):
        self.capacity = capacity  # Количество сохраняемых состояний
        self.device = device  # Устройство для CLIP
        self.memory = deque(maxlen=capacity)  # Дек для хранения (изображение, v, w)
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)  # CLIP модель
        self.similarity_threshold = similarity_threshold  # Порог схожести изображений
        self._initialize_black_images()  # Инициализация черных изображений
    
    def _initialize_black_images(self):
        """Заполняет память черными изображениями при инициализации."""
        black_image = Image.new("RGB", (400, 400), (0, 0, 0))
        for _ in range(self.capacity):
            self.memory.append((black_image, 0.0, 0.0))
    
    def preprocess_image(self, image):
        """Препроцессинг изображения перед передачей в CLIP"""
        return self.preprocess(image).unsqueeze(0).to(self.device)
    
    def compute_clip_feature(self, image):
        """Извлекает признак изображения с помощью CLIP"""
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def is_similar(self, new_image):
        """Проверяет, есть ли в памяти похожее изображение"""
        new_image_clip = self.compute_clip_feature(self.preprocess_image(new_image))
        
        for img, _, _ in self.memory:
            img_clip = self.compute_clip_feature(self.preprocess_image(img))
            similarity = torch.cosine_similarity(new_image_clip, img_clip)
            if similarity.item() > self.similarity_threshold:
                print("sim image ", similarity.item())
                return True  # Похожее изображение уже есть в памяти
        
        return False
    
    def add(self, image, linear_velocity, angular_velocity):
        """Добавляет новое изображение и связанные параметры в память"""
        if not self.is_similar(image):
            self.memory.append((image, linear_velocity, angular_velocity))
    
    def get_memory(self):
        """Возвращает текущую память в виде списка"""
        return self.get_embedding()
    
    def get_embedding(self, method="concat"):
        """
        Возвращает эмбеддинг от всего набора изображений в памяти.
        
        :param method: Метод объединения эмбеддингов. Возможные значения: "mean" (усреднение), "concat" (конкатенация).
        :return: Тензор с общим эмбеддингом.
        """
        if len(self.memory) == 0:
            return None

        # Получаем эмбеддинги для всех изображений в памяти
        embeddings = []
        for img, v, w in self.memory:
            img_clip = self.compute_clip_feature(self.preprocess_image(img))
            embeddings.append(img_clip)#[img_clip,v,w])
        
        if method == "concat":
            # Конкатенация эмбеддингов
            combined_embedding = torch.cat(embeddings, dim=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return combined_embedding

    
    def save_memory_as_grid(self, save_path="/home/kit/.local/share/ov/pkg/isaac-sim-4.1.0/standalone_examples/Aloha_graph/memory_grid.jpg"):
        """Создаёт и сохраняет сетку изображений"""
        if len(self.memory) == 0:
            print("Память пуста, нечего сохранять.")
            return
        
        images = [img for img, _, _ in self.memory]
        grid_size = int(np.ceil(np.sqrt(len(images))))
        
        transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
        images = [transform(img) for img in images]
        grid = make_grid(images, nrow=grid_size)
        save_image(grid, save_path)
        # save_image(make_grid(img[0], nrows = 2), '/home/kit/.local/share/ov/pkg/isaac-sim-2023.1.1/standalone_examples/Aloha_graph/Aloha/img/memory.png')
        print(f"Сетка сохранена в {save_path}")

memory = ImageMemoryManager()
img = Image.new("RGB", (400, 400), (255, 255, 255))
memory.add(img,0,0)
for i in range(10):
    img = Image.new("RGB", (400, 400), (i*10, i*10, i*10))
    memory.add(img,1,1)
memory.save_memory_as_grid()
emb = memory.get_embedding()
print(len(emb[0]))
print(memory.get_memory())
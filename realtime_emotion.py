import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.net import Net

class EmotionDetector:
    def __init__(self, model_path, device='cpu'):
        self.model_path = model_path
        self.device = device
        self.model = self._load_model(self.model_path)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.emotions = {
            0: 'Angry', 1: 'Disgust', 2: 'Fear',
            3: 'Happy', 4: 'Sad', 5: 'Surprise',
            6: 'Neutral'
        }
        
    def _load_model(self, model_path):
        model = Net(num_classes=7)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    def preprocess_face(self, frame):
        gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_face = Image.fromarray(gray_face)
        tensor_face = self.transform(pil_face)
        return tensor_face.unsqueeze(0).to(self.device)

    def detect_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                tensor_face = self.preprocess_face(face_img)
                with torch.no_grad():
                    output = self.model(tensor_face)
                    emotion_idx = torch.argmax(output).item()
                    emotion = self.emotions[emotion_idx]
                    confidence = torch.softmax(output, dim=1)[0][emotion_idx].item()
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        return frame

def main():
    detector = EmotionDetector('models/saved_models/best_model.pth')
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = detector.detect_emotion(frame)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

import cv2
import numpy as np
import onnxruntime as ort

class ReidFeatureExtractor:
    def __init__(self, model_path='reid_osnet_x025_market1501.onnx'):
        self.roi_width = 64
        self.roi_height = 128

        # means and stds based on ImageNet
        self.roi_means = np.array([0.485, 0.456, 0.406])
        self.roi_stds = np.array([0.229, 0.224, 0.225])

        self.session = ort.InferenceSession(model_path)

    def preprocess_patch(self, im_crops):
        # preprocessing given in the subject
        roi_input = cv2.resize(im_crops, (self.roi_width, self.roi_height))
        roi_input = cv2.cvtColor(roi_input, cv2.COLOR_BGR2RGB)
        roi_input = (np.asarray(roi_input).astype(np.float32) - self.roi_means) / self.roi_stds
        roi_input = np.moveaxis(roi_input, -1, 0)
        object_patch = roi_input.astype('float32')
        return object_patch

    def extract_features(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        patch = frame[y: y + h, x: x + w]

        if patch.size == 0:
            return None

        preprocessed_patch = self.preprocess_patch(patch)
        input_name = self.session.get_inputs()[0].name
        features = self.session.run(None, {input_name: preprocessed_patch[np.newaxis, ...]})[0]
        return features.squeeze()

    def compute_similarity(self, features1, features2, metric='cosine'):
        if metric == 'cosine':
            return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        elif metric == 'euclidean':
            return 1 / (1 + np.linalg.norm(features1 - features2))
        else:
            raise Exception
from libSym.FeatureExtractor import FeatureExtractor
from libSym.ImageLoader import ImageLoader

imageLoader = ImageLoader(data_path="../data")

train_images = imageLoader.get_train_image_matrices()
print str(train_images)

feature_extractor = FeatureExtractor(7)
features = feature_extractor.extract_features_multi(train_images, verbose=True)



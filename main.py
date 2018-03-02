import data_preparation as dp
from metric import score_image

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
path = r'C:\Users\omri\Personal\Kaggle\Project'
TRAIN_PATH = os.path.join(path, 'stage1_train')  # '"C:\Users\omri\Personal\Kaggle\Project\stage1_train/'
TEST_PATH = '"C:\Users\omri\Personal\Kaggle\Project\stage1_test'

# Read the files and prepare data
X_train, Y_train, X_test = dp.read_images(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH, TEST_PATH)

# Train

# Predict

# Submit
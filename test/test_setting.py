class TestSetting:

    def __init__(self):
        self.network_address = 'example/classification/Vgg16_network.json'
        self.input_layer_name = "input_8"
        self.number_of_item = 91
        self.properties_root_name = 'InputLayer'
        self.properties_length = 4
        self.model_address = 'example/classification/Vgg16.h5'
        self.image_name = 'example/classification/cat_dog.png'
        self.mapping_file_path = "example/classification/imagenet_class_index.json"
        self.prediction_count = 5
        self.prediction_vgg16_cat_dog_0 = 'boxer 0.42'
        self.prediction_vgg16_cat_dog_3 = 'tiger 0.05'

import cv2
import numpy as np
import sys
import tensorflow as tf

IMG_WIDTH = 30
IMG_HEIGHT = 30

# Ánh xạ từ nhãn của tập dữ liệu sang tên của biển báo giao thông
label_to_name = {
    0: "Giới hạn tốc độ (20km/h)",
    1: "Giới hạn tốc độ (30km/h)",
    2: "Giới hạn tốc độ (50km/h)",
    3: "Giới hạn tốc độ (60km/h)",
    4: "Giới hạn tốc độ (70km/h)",
    5: "Giới hạn tốc độ (80km/h)",
    6: "Hết giới hạn tốc độ (80km/h)",
    7: "Giới hạn tốc độ (100km/h)",
    8: "Giới hạn tốc độ (120km/h)",
    9: "Cấm vượt",
    10: "Cấm vượt đối với xe trên 3.5 tấn",
    11: "Quyền ưu tiên tại ngã tư tiếp theo",
    12: "Đường ưu tiên",
    13: "Nhường đường",
    14: "Dừng lại",
    15: "Cấm các loại xe",
    16: "Cấm xe trên 3.5 tấn",
    17: "Cấm vào",
    18: "Cảnh báo chung",
    19: "Đường cong nguy hiểm bên trái",
    20: "Đường cong nguy hiểm bên phải",
    21: "Đường cong kép",
    22: "Đường gồ ghề",
    23: "Đường trơn",
    24: "Đường hẹp bên phải",
    25: "Công trường",
    26: "Tín hiệu giao thông",
    27: "Người đi bộ",
    28: "Trẻ em qua đường",
    29: "Xe đạp qua đường",
    30: "Cẩn thận băng tuyết",
    31: "Động vật hoang dã qua đường",
    32: "Hết tất cả các giới hạn tốc độ và cấm vượt",
    33: "Rẽ phải phía trước",
    34: "Rẽ trái phía trước",
    35: "Chỉ đi thẳng",
    36: "Đi thẳng hoặc rẽ phải",
    37: "Đi thẳng hoặc rẽ trái",
    38: "Đi bên phải",
    39: "Đi bên trái",
    40: "Vòng xuyến bắt buộc",
    41: "Hết cấm vượt",
    42: "Hết cấm vượt đối với xe trên 3.5 tấn"
}

def get_traffic_sign_name(label):
    return label_to_name.get(label, "Unknown sign")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py model.h5")

    model_path = sys.argv[1]
    model = tf.keras.models.load_model(model_path)

    while True:
        image_path = input("Enter the path to the image file (or 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break

        image = load_image(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        prediction = model.predict(np.array([image]))
        predicted_category = np.argmax(prediction)

        # Get the traffic sign name
        traffic_sign_name = get_traffic_sign_name(predicted_category)
        # Print the predicted category and the confidence
        print(f"Predicted category: {predicted_category} - {traffic_sign_name}")
        print(f"Prediction confidence: {np.round(prediction[0], 2)}")

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0
    return image

if __name__ == "__main__":
    main()
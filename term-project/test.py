import cv2
import numpy as np
import os

def process_xray_image(input_path, output_folder):
    # อ่านภาพ
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # ปรับแต่งภาพด้วยการเพิ่ม contrast และลด noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # หาคอนทัวร์
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # เลือกคอนทัวร์ที่มีลักษณะเหมือนกระดูกสันหลัง
    spine_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]  # กรองตามพื้นที่
    
    # สร้างหน้ากากสำหรับส่วนกระดูกสันหลัง
    mask = np.zeros_like(image)
    cv2.drawContours(mask, spine_contours, -1, (255), thickness=cv2.FILLED)
    
    # ตัดส่วนกระดูกออกมา
    spine_only = cv2.bitwise_and(image, image, mask=mask)
    
    # แยกกระดูกเป็นข้อ
    for i, cnt in enumerate(spine_contours):
        x, y, w, h = cv2.boundingRect(cnt)
        vertebra = spine_only[y:y+h, x:x+w]
        vertebra = cv2.resize(vertebra, (128, 128))  # ปรับขนาดให้เหมาะสม
        
        # ตั้งชื่อไฟล์ตามลำดับ
        output_path = os.path.join(output_folder, f'vertebra_{i+1}.jpg')
        cv2.imwrite(output_path, vertebra)

# เรียกใช้งานฟังก์ชัน
input_image_path = 'input/Image-1.jpeg'
output_directory = 'input'

process_xray_image(input_image_path, output_directory)


import cv2
import numpy as np
import pyodbc
import os


faceDetect = cv2.CascadeClassifier(r"C:\Users\user\Downloads\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0) # to detect faces

"""
also need to add to record data in excel fil
"""
def insertupdate(id, name, age):
    conn = pyodbc.connect("Driver={SQL Server};"
                          "Server=DESKTOP-3QJN7S3;"  # Server name
                          "Database=Test;"  # selected database
                          "Trusted_Connection=yes;")
    cursor = conn.cursor()

    cursor.execute('select * from Test.dbo.FaceData where id = ?', (id,))
    isrecordexists = cursor.fetchone()

    if isrecordexists:
        conn.execute('update Test.dbo.FaceData set name = ?, age = ? where id = ?',
                     (name, age, id)
                     )

    else:
        conn.execute('''insert into Test.dbo.FaceData (id, name, age) values(?,?,?)''',
                     (id, name, age)
                     )
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

id = int(input("Enter ID"))
name = input("Enter Name")
age = int(input("Enter age"))

insertupdate(id, name, age)

sampleNum = 0
dataset_path = r"D:\Data\Python Programming\python project\computervision\dataset"

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        face_img = gray[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(dataset_path, f"{id}.{sampleNum}.jpg"), face_img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)

    cv2.imshow("Face", img) # show faces detected in web camera
    cv2.waitKey(1)

    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()

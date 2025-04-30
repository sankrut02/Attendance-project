import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime, time
import mediapipe as mp
import math
import smtplib
from email.message import EmailMessage

path = 'Face Detection/images'
images = []
classnames = []
mylist = os.listdir(path)

for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])

# Blinking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)


def calculate_ear(landmarks, eye_indices):
    def euclidean(p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    vertical1 = euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    vertical2 = euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    horizontal = euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def imgencoding(images):
    conlist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        con = face_recognition.face_encodings(img)[0]
        conlist.append(con)
    return conlist


def markAttendance(name):
    with open('Face Detection/Attendance.csv', 'r+') as f:
        mydatalist = f.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])

        if name not in namelist:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtstring}')
        print(mydatalist)


def AttendanceCheck(name):
    with open('Face Detection/Attendance.csv', 'r') as k:
        mydatalist = k.readlines()
        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])

        if name in namelist:
            return False
        else:
            return True


def update_student_attendance(name):
    filename = 'Face Detection/Students.csv'
    with open(filename, 'r') as f:
        lines = f.readlines()

    updated = False
    for i, line in enumerate(lines):
        entry = line.strip().split(',')
        if entry[0] == name:
            entry[1] = 'Present'
            lines[i] = ','.join(entry) + '\n'
            updated = True
            break

    if not updated:  
        with open(filename, 'a') as f:
            f.write(f'{name},Present,\n') 

    else:
        with open(filename, 'w') as f:
            f.writelines(lines)


cutofftime=time(23,24)
def within_cutoff(cutofftime):
    current_time = datetime.now().time()
    if current_time <= cutofftime:
        state = 0
    else:
        state = 1
    return state


convlistknown = imgencoding(images)
print("Conversion Complete")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgsmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgsmall = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2RGB)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    # Blinking check
    blink_detected = False
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.21:
                blink_detected = True

            else:
                blink_detected = False

    facesCurFrame = face_recognition.face_locations(imgsmall)
    convCurFrame = face_recognition.face_encodings(imgsmall, facesCurFrame)

    for convface, faceloc in zip(convCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(convlistknown, convface)
        faceDis = face_recognition.face_distance(convlistknown, convface)
        print(faceDis)
        matchindex = np.argmin(faceDis)

        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            state = within_cutoff(cutofftime)

            if state == 0:

                cv2.putText(img, "Attendance Approved", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, str(datetime.now().time()), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 255), 2)
                if blink_detected:
                    cv2.putText(img, "Blink Detected", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    markAttendance(name)
                    update_student_attendance(name)

                else:
                    cv2.putText(img, "Blink Not Detected. Please Blink", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (0, 0, 255), 2)

            else:
                cv2.putText(img, "Attendance NOT Approved", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(img, str(datetime.now().time()), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 255), 2)

            if blink_detected:
                blink_detected = AttendanceCheck(name)

    cur_time = datetime.now().time()
    
    if cutofftime < cur_time:
        break
    cv2.imshow('webcam', img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()


def update_absent_status():
    """Updates the Students.csv file to mark students as Absent if they were not marked Present during the attendance session."""
    filename = 'Face Detection/Students.csv'
    with open(filename, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        entry = line.strip().split(',')
        if len(entry) > 1 and entry[1] == 'Present':
            updated_lines.append(line)  
        elif len(entry) > 1:
            entry[1] = 'Absent'  
            updated_lines.append(','.join(entry) + '\n')
        else:
            updated_lines.append(line)
    with open(filename, 'w') as f:
        f.writelines(updated_lines)



def get_absent_emails():
    """Retrieves the emails of students marked as Absent in the Students.csv file."""
    filename = 'Face Detection/Students.csv'
    absent_emails = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        entry = line.strip().split(',')
        if len(entry) > 1 and entry[1] == 'Absent' and len(entry) > 2:  
            absent_emails.append(entry[2])
    return absent_emails



def send_absent_emails(email_absentees):
   
    sender_email = "phoenixbanu@gmail.com" 
    sender_password = "jdoe mtgn ofbg tfmh"  

   
    subject = "Absentee Notice"
    body = """
Dear Student,

Our records indicate that you were absent today. 
Please ensure your attendance in upcoming sessions.

Regards,
Praveen Sir,'
AIML A Co-ordinator
"""

    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
    except Exception as e:
        print(f"Failed to connect to SMTP server: {e}")
        return  
    
    for recipient in email_absentees:
        msg = EmailMessage()
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = subject
        msg.set_content(body)

        try:
            server.send_message(msg)
            print(f"Email sent to {recipient}")
        except Exception as e:
            print(f"Failed to send email to {recipient}: {e}")

    server.quit()


if __name__ == '__main__':
    update_absent_status()
    absent_emails = get_absent_emails()
    send_absent_emails(absent_emails)
